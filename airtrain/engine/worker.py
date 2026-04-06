"""Worker node for AirTrain distributed training.

Connects to a coordinator, receives model weights, trains locally,
and sends pseudo-gradients back for aggregation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from airtrain.compat import MLX_AVAILABLE, require_mlx
from airtrain.config import NetworkConfig, PeerRole, TrainingConfig
from airtrain.discovery.peer import get_local_peer_info
from airtrain.network.compression import compress_gradients, decompress_gradients
from airtrain.network.protocol import Message, MessageType
from airtrain.network.transport import TransportClient

logger = logging.getLogger(__name__)


async def run_worker(address: str, network_config: NetworkConfig):
    """Run a worker node.

    Connects to the coordinator, receives weights, trains locally,
    and sends pseudo-gradients.
    """
    require_mlx()
    import mlx.core as mx

    from airtrain.engine.diloco import DiLoCoEngine
    from airtrain.engine.trainer import BaseTrainer
    from airtrain.models.registry import get_model

    peer_info = get_local_peer_info(role=PeerRole.WORKER)

    # Discover coordinator
    host, port = await _resolve_coordinator(address, network_config)

    # Connect
    client = TransportClient(peer_info.peer_id)

    # State for sync coordination
    sync_data: dict = {}
    sync_event = asyncio.Event()
    weights_event = asyncio.Event()
    model_ready = asyncio.Event()
    training_config_holder: dict = {}

    def on_message(msg: Message):
        if msg.msg_type == MessageType.SYNC_REQUEST:
            # Coordinator wants us to sync — store the original weights and train
            sync_data["original_weights"] = decompress_gradients(msg.payload)
            sync_data["global_step"] = msg.metadata.get("global_step", 0)
            sync_data["inner_steps"] = msg.metadata.get("inner_steps", 500)
            training_config_holder["model_name"] = msg.metadata.get("model_name", "gpt2-tiny")
            sync_event.set()

        elif msg.msg_type == MessageType.MODEL_WEIGHTS:
            # Received updated weights after outer step
            sync_data["new_weights"] = decompress_gradients(msg.payload)
            weights_event.set()

    client.on_message = on_message

    await client.connect(host, port)
    print(f"Connected to coordinator at {host}:{port}")
    print(f"Peer ID: {peer_info.peer_id}")
    print(f"Chip: {peer_info.chip}")
    print(f"Waiting for training to begin...\n")

    # Wait for first sync request to know what model to create
    model = None
    trainer = None
    diloco = None

    try:
        while client.connected:
            # Wait for sync request from coordinator
            sync_event.clear()
            await sync_event.wait()

            inner_steps = sync_data.get("inner_steps", 500)
            global_step = sync_data.get("global_step", 0)
            original_weights_np = sync_data["original_weights"]

            # Lazy model creation on first sync
            if model is None:
                model_name = training_config_holder.get("model_name", "gpt2-tiny")
                model = get_model(model_name)
                config = TrainingConfig(model_name=model_name)
                trainer = BaseTrainer(model, config)
                diloco = DiLoCoEngine(config.diloco)
                logger.info("Created model: %s", model_name)

            # Load the original weights
            original_mx = diloco.numpy_to_params(original_weights_np)
            trainer.set_parameters(original_mx)
            diloco.snapshot_params(original_mx)

            # Run inner training loop
            print(f"Training inner loop ({inner_steps} steps)...")
            trainer.reset_metrics()
            for step in range(inner_steps):
                # Synthetic data (replace with real data loader)
                batch_x = mx.random.randint(0, 1000, (8, 512))
                batch_y = mx.random.randint(0, 1000, (8, 512))
                loss = trainer.train_step(batch_x, batch_y)

                if (step + 1) % 50 == 0:
                    print(f"  [inner {step+1}/{inner_steps}] loss={loss:.4f}", flush=True)

            # Compute and send pseudo-gradients
            current_params = trainer.get_parameters()
            pseudo_grads = diloco.compute_pseudo_gradients(
                diloco.original_params, current_params
            )
            grads_np = diloco.params_to_numpy(pseudo_grads)
            compressed = compress_gradients(grads_np)

            await client.send(
                Message(
                    msg_type=MessageType.SYNC_GRADIENTS,
                    sender_id=peer_info.peer_id,
                    payload=compressed,
                    metadata={"global_step": global_step},
                )
            )
            print(f"Sent pseudo-gradients ({len(compressed)/1e6:.1f} MB)")

            # Wait for updated weights
            weights_event.clear()
            try:
                await asyncio.wait_for(weights_event.wait(), timeout=60.0)
                new_weights_np = sync_data["new_weights"]
                new_weights_mx = diloco.numpy_to_params(new_weights_np)
                trainer.set_parameters(new_weights_mx)
                print(f"Received updated weights (step {global_step})\n")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for updated weights")

    except KeyboardInterrupt:
        print("\nWorker stopping...")
    except asyncio.CancelledError:
        pass
    finally:
        await client.disconnect()


async def _resolve_coordinator(
    address: str, config: NetworkConfig
) -> tuple[str, int]:
    """Resolve coordinator address. 'auto' uses mDNS discovery."""
    if address == "auto":
        from airtrain.discovery.mdns import discover_peers

        print("Scanning for AirTrain coordinators on local network...")
        peers = discover_peers(timeout=5.0)

        if not peers:
            raise ConnectionError(
                "No AirTrain coordinators found on local network. "
                "Make sure the coordinator is running with 'airtrain start'."
            )

        # Pick the first coordinator found
        peer = peers[0]
        print(f"Found coordinator: {peer.hostname} at {peer.ip_address}:{peer.port}")
        return peer.ip_address, peer.port

    # Parse host:port
    if ":" in address:
        host, port_str = address.rsplit(":", 1)
        return host, int(port_str)

    return address, config.listen_port
