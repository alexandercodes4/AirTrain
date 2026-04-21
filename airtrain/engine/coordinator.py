"""Coordinator node for AirTrain distributed training.

The coordinator manages the training loop, collects pseudo-gradients
from workers, applies the outer optimizer step, and broadcasts updates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from airtrain.compat import MLX_AVAILABLE, require_mlx
from airtrain.config import (
    CheckpointMeta,
    MarketplaceConfig,
    NetworkConfig,
    PeerRole,
    PeerStatus,
    TrainingConfig,
)
from airtrain.engine.marketplace import GradientMarketplace
from airtrain.discovery.peer import PeerManager, get_local_peer_info
from airtrain.network.compression import compress_gradients, decompress_gradients
from airtrain.network.protocol import Message, MessageType
from airtrain.network.transport import TransportServer

logger = logging.getLogger(__name__)


async def run_coordinator(
    training_config: TrainingConfig,
    network_config: NetworkConfig,
    resume_path: Optional[str] = None,
):
    """Run the coordinator node.

    Starts the TCP server, registers mDNS, and orchestrates
    the DiLoCo training loop.
    """
    require_mlx()
    import mlx.core as mx

    from airtrain.engine.checkpoint import load_checkpoint, save_checkpoint
    from airtrain.engine.diloco import DiLoCoEngine
    from airtrain.engine.trainer import BaseTrainer
    from airtrain.models.registry import get_model

    # Setup
    peer_info = get_local_peer_info(port=network_config.listen_port, role=PeerRole.COORDINATOR)
    peer_manager = PeerManager()
    peer_manager.add_peer(peer_info)

    # Create model
    model = get_model(training_config.model_name)
    trainer = BaseTrainer(model, training_config)
    diloco = DiLoCoEngine(training_config.diloco)
    marketplace = GradientMarketplace(MarketplaceConfig())

    param_count = sum(p.size for p in model.parameters().values())
    logger.info("Model %s: %d parameters", training_config.model_name, param_count)

    # Resume from checkpoint if specified
    global_step = 0
    if resume_path:
        params, _, meta = load_checkpoint(resume_path)
        mlx_params = {k: mx.array(v) for k, v in params.items()}
        trainer.set_parameters(mlx_params)
        global_step = meta.global_step
        logger.info("Resumed from step %d", global_step)

    # Gradient collection state
    pending_gradients: dict[str, dict[str, np.ndarray]] = {}
    sync_event = asyncio.Event()
    workers_in_round: set[str] = set()

    # Transport callbacks
    async def on_connect(peer_id: str):
        logger.info("Worker connected: %s", peer_id)
        workers_in_round.add(peer_id)

    async def on_disconnect(peer_id: str):
        logger.info("Worker disconnected: %s", peer_id)
        peer_manager.remove_peer(peer_id)
        workers_in_round.discard(peer_id)

    async def on_message(peer_id: str, msg: Message):
        if msg.msg_type == MessageType.SYNC_GRADIENTS:
            grads = decompress_gradients(msg.payload)
            pending_gradients[peer_id] = grads
            logger.info(
                "Received gradients from %s (%d params)",
                peer_id,
                len(grads),
            )
            # Check if all workers have reported
            if len(pending_gradients) >= len(workers_in_round):
                sync_event.set()

        elif msg.msg_type == MessageType.HANDSHAKE:
            meta = msg.metadata
            worker_peer = get_local_peer_info(role=PeerRole.WORKER)
            worker_peer.peer_id = peer_id
            worker_peer.chip = meta.get("chip", "")
            peer_manager.add_peer(worker_peer)

    # Start server
    server = TransportServer(network_config)
    server.on_connect = on_connect
    server.on_disconnect = on_disconnect
    server.on_message = on_message
    await server.start()

    # Register mDNS
    zc = None
    if network_config.use_mdns:
        try:
            from airtrain.discovery.mdns import register_service

            zc, _ = register_service(peer_info)
            logger.info("mDNS service registered")
        except Exception as e:
            logger.warning("Failed to register mDNS: %s", e)

    # Start dashboard if enabled
    dashboard_task = None
    if training_config.enable_dashboard:
        try:
            from airtrain.dashboard.app import create_app, run_dashboard

            dashboard_task = asyncio.create_task(
                run_dashboard(training_config.dashboard_port)
            )
            logger.info("Dashboard at http://localhost:%d", training_config.dashboard_port)
        except Exception as e:
            logger.warning("Failed to start dashboard: %s", e)

    print(f"\nCoordinator ready on port {network_config.listen_port}")
    print(f"Workers can join with: airtrain join auto\n")

    # Training loop
    start_time = time.time()
    inner_steps = training_config.diloco.inner_steps

    try:
        while global_step < training_config.max_steps:
            # Snapshot parameters before inner loop
            current_params = trainer.get_parameters()
            diloco.snapshot_params(current_params)

            # Generate synthetic data for demo (replace with real data loader)
            trainer.reset_metrics()
            for step in range(inner_steps):
                batch_x = mx.random.randint(0, 1000, (training_config.batch_size, training_config.seq_length))
                batch_y = mx.random.randint(0, 1000, (training_config.batch_size, training_config.seq_length))
                loss = trainer.train_step(batch_x, batch_y)

                if (step + 1) % training_config.log_every == 0:
                    print(
                        f"  [inner {step+1}/{inner_steps}] loss={loss:.4f}",
                        flush=True,
                    )

            global_step += inner_steps

            # Compute coordinator's pseudo-gradients
            updated_params = trainer.get_parameters()
            coord_grads = diloco.compute_pseudo_gradients(
                diloco.original_params, updated_params
            )
            coord_grads_np = diloco.params_to_numpy(coord_grads)

            # Broadcast sync request to workers
            if workers_in_round:
                # Send current weights so workers can sync
                weights_data = compress_gradients(diloco.params_to_numpy(diloco.original_params))
                await server.broadcast(
                    Message(
                        msg_type=MessageType.SYNC_REQUEST,
                        sender_id=peer_info.peer_id,
                        payload=weights_data,
                        metadata={"global_step": global_step, "inner_steps": inner_steps},
                    )
                )

                # Wait for worker gradients (with timeout)
                pending_gradients.clear()
                sync_event.clear()
                try:
                    await asyncio.wait_for(sync_event.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Sync timeout: got %d/%d workers",
                        len(pending_gradients),
                        len(workers_in_round),
                    )

            # Collect all pseudo-gradients (coordinator + workers)
            all_peer_ids = [peer_info.peer_id]
            all_grads_np = [coord_grads_np]
            for pid, grads in pending_gradients.items():
                all_peer_ids.append(pid)
                all_grads_np.append(grads)

            # Score gradients via Marketplace
            prev_loss = trainer.avg_loss
            grads_by_peer = dict(zip(all_peer_ids, all_grads_np))
            mp_weights = marketplace.score_gradients(grads_by_peer, diloco.outer_step)
            weight_list = [mp_weights[pid] for pid in all_peer_ids]

            # Convert to MLX and apply outer step (weighted)
            all_grads_mx = [diloco.numpy_to_params(g) for g in all_grads_np]
            new_params = diloco.apply_outer_step(all_grads_mx, weights=weight_list)

            # Update model
            trainer.set_parameters(new_params)

            # Update marketplace history with loss delta
            new_loss = trainer.avg_loss
            loss_delta = new_loss - prev_loss
            for pid in all_peer_ids:
                marketplace.update_history(pid, mp_weights[pid], loss_delta)

            # Log marketplace rankings
            rankings = marketplace.get_rankings()
            if rankings:
                rank_str = " | ".join(
                    f"#{s.rank} {s.peer_id[:8]} w={s.weight:.3f}"
                    for s in rankings[:5]
                )
                logger.info("Marketplace: %s", rank_str)

            # Broadcast updated weights to workers
            if workers_in_round:
                new_weights_data = compress_gradients(diloco.params_to_numpy(new_params))
                await server.broadcast(
                    Message(
                        msg_type=MessageType.MODEL_WEIGHTS,
                        sender_id=peer_info.peer_id,
                        payload=new_weights_data,
                        metadata={
                            "global_step": global_step,
                            "marketplace_scores": {
                                s.peer_id: {"weight": s.weight, "rank": s.rank}
                                for s in rankings
                            },
                        },
                    )
                )

            elapsed = time.time() - start_time
            print(
                f"\n[step {global_step}] loss={trainer.avg_loss:.4f} "
                f"peers={peer_manager.active_count} "
                f"elapsed={elapsed:.0f}s\n",
                flush=True,
            )

            # Checkpoint
            if global_step % training_config.checkpoint_every == 0:
                ckpt_path = Path(training_config.checkpoint_dir) / f"step-{global_step}"
                meta = CheckpointMeta(
                    model_name=training_config.model_name,
                    global_step=global_step,
                    loss=trainer.avg_loss,
                    contributors=[peer_info.hostname],
                    total_compute_hours=elapsed / 3600,
                )
                params_np = diloco.params_to_numpy(new_params)
                save_checkpoint(ckpt_path, params_np, None, meta)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        ckpt_path = Path(training_config.checkpoint_dir) / f"step-{global_step}"
        meta = CheckpointMeta(
            model_name=training_config.model_name,
            global_step=global_step,
            loss=trainer.avg_loss,
            total_compute_hours=(time.time() - start_time) / 3600,
        )
        params_np = diloco.params_to_numpy(trainer.get_parameters())
        save_checkpoint(ckpt_path, params_np, None, meta)
    finally:
        await server.stop()
        if zc:
            zc.close()
        if dashboard_task:
            dashboard_task.cancel()
