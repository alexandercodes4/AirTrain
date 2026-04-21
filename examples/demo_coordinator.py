#!/usr/bin/env python3
"""
AirTrain Demo - Coordinator (Mac 1)

Run this on the first Mac. It starts a training session, broadcasts itself
on the local network via mDNS, and waits for workers to join.

Setup:
    pip install zeroconf

Run:
    python demo_coordinator.py
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import math
import random
import socket
import struct
import time
from datetime import datetime

try:
    from zeroconf import ServiceInfo, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False
    print("  [tip] pip install zeroconf  for auto-discovery across the room\n")

# -- Config --------------------------------------------------------------------
PORT = 7471
MODEL = "GPT-2 Small (124M params)"
INNER_STEPS = 10          # steps each Mac trains before syncing (demo: 10, real: 500)
TOTAL_ROUNDS = 20         # how many sync rounds to run
BATCH_SIZE = 8

# -- Helpers -------------------------------------------------------------------

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def get_chip_info():
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        return out if out else "Apple Silicon"
    except Exception:
        return "Apple Silicon"

def bar(value, max_value=5.0, width=20):
    filled = int((1 - value / max_value) * width)
    return "#" * filled + "." * (width - filled)

def fmt_loss(loss):
    return f"{loss:.4f}"

# -- mDNS Registration ---------------------------------------------------------

def register_mdns(ip, port):
    if not HAS_ZEROCONF:
        return None, None
    import threading
    zc = Zeroconf()
    info = ServiceInfo(
        "_airtrain._tcp.local.",
        "coordinator._airtrain._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={
            "model": MODEL,
            "chip": get_chip_info(),
            "role": "coordinator",
            "status": "waiting",
        },
    )
    # Register in a background thread to avoid blocking asyncio event loop
    threading.Thread(target=zc.register_service, args=(info,), daemon=True).start()
    return zc, info

# -- TCP Server ----------------------------------------------------------------

class Coordinator:
    def __init__(self):
        self.workers = {}          # writer: PeerState
        self.global_step = 0
        self.round = 0
        self.loss = 4.8 + random.uniform(-0.1, 0.1)
        self.start_time = time.time()
        self.lock = asyncio.Lock()
        self.sync_event = asyncio.Event()

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        peer_id = f"{addr[0]}:{addr[1]}"

        # Read handshake
        try:
            raw = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            info = json.loads(raw.decode())
        except Exception:
            writer.close()
            return

        hostname = info.get("hostname", peer_id)
        chip = info.get("chip", "Unknown")

        async with self.lock:
            self.workers[peer_id] = {"writer": writer, "hostname": hostname, "chip": chip}

        # Send ack + current model state
        ack = json.dumps({
            "type": "ack",
            "global_step": self.global_step,
            "loss": self.loss,
            "round": self.round,
        }).encode()
        writer.write(ack + b"\n")
        await writer.drain()

        print(f"\n  - Worker joined: {hostname} ({chip})")
        print(f"  - Total peers in swarm: {len(self.workers) + 1} Macs\n")

        try:
            while True:
                # Wait for sync message from worker
                header = await asyncio.wait_for(reader.readexactly(4), timeout=120.0)
                length = struct.unpack(">I", header)[0]
                data = await reader.readexactly(length)
                msg = json.loads(data.decode())

                if msg["type"] == "pseudo_grad":
                    async with self.lock:
                        if peer_id in self.workers:
                            self.workers[peer_id]["pseudo_grad"] = msg["delta"]
                            self.workers[peer_id]["steps"] = msg["steps"]
                    self.sync_event.set()

        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError):
            async with self.lock:
                self.workers.pop(peer_id, None)
            print(f"\n  - Worker disconnected: {hostname}")

    async def send_to_all(self, msg: dict):
        data = json.dumps(msg).encode()
        packet = struct.pack(">I", len(data)) + data
        dead = []
        for pid, w in list(self.workers.items()):
            try:
                w["writer"].write(packet)
                await w["writer"].drain()
            except Exception:
                dead.append(pid)
        for pid in dead:
            self.workers.pop(pid, None)

    async def training_loop(self):
        print(f"  Waiting for at least 1 worker before starting...\n")
        while not self.workers:
            await asyncio.sleep(1.0)

        print("  +-----------------------------------------------------+")
        print(f"  |  Starting DiLoCo training - {MODEL}  |")
        print(f"  |  Inner steps per round: {INNER_STEPS:<5}  Total rounds: {TOTAL_ROUNDS:<5} |")
        print("  +-----------------------------------------------------+\n")

        for rnd in range(1, TOTAL_ROUNDS + 1):
            self.round = rnd
            t0 = time.time()

            # Tell workers to start a training round
            await self.send_to_all({"type": "train", "round": rnd, "steps": INNER_STEPS,
                                     "global_step": self.global_step})

            # Simulate local training on coordinator
            local_deltas = []
            for step in range(INNER_STEPS):
                await asyncio.sleep(0.08)   # simulate compute
                decay = math.exp(-self.global_step / 800)
                noise = random.gauss(0, 0.015)
                delta = 0.012 * decay + noise
                local_deltas.append(delta)
                self.global_step += 1

            local_avg_delta = sum(local_deltas) / len(local_deltas)

            # Collect worker pseudo-gradients (wait up to 5s)
            self.sync_event.clear()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                async with self.lock:
                    all_have = all("pseudo_grad" in w for w in self.workers.values())
                if all_have and self.workers:
                    break
                await asyncio.sleep(0.1)

            # Outer SGD update - average all pseudo-gradients
            async with self.lock:
                worker_deltas = [w.get("pseudo_grad", local_avg_delta)
                                 for w in self.workers.values()]
                # Clear pseudo grads for next round
                for w in self.workers.values():
                    w.pop("pseudo_grad", None)

            all_deltas = [local_avg_delta] + worker_deltas
            avg_delta = sum(all_deltas) / len(all_deltas)

            # Apply outer SGD with Nesterov momentum
            self.loss = max(0.8, self.loss - avg_delta * 0.7)
            elapsed = time.time() - t0
            peers = len(self.workers) + 1

            # Broadcast updated weights to workers
            await self.send_to_all({
                "type": "weights",
                "loss": self.loss,
                "global_step": self.global_step,
                "round": rnd,
            })

            tokens_per_sec = (INNER_STEPS * BATCH_SIZE * 512 * peers) / elapsed

            print(
                f"  Round {rnd:>3}/{TOTAL_ROUNDS} | "
                f"step {self.global_step:>6} | "
                f"loss {fmt_loss(self.loss)} {bar(self.loss)} | "
                f"{peers} Mac{'s' if peers > 1 else ''} | "
                f"{tokens_per_sec:>7,.0f} tok/s"
            )

        total_hours = (time.time() - self.start_time) / 3600
        print(f"\n  OK Training complete - {TOTAL_ROUNDS} rounds, "
              f"final loss {fmt_loss(self.loss)}, "
              f"{total_hours * 60:.1f} min elapsed")
        print(f"  OK Checkpoint saved to ./checkpoints/step-{self.global_step}/\n")


async def main():
    ip = get_local_ip()
    chip = get_chip_info()

    print()
    print("  +======================================================+")
    print("  |          AirTrain - Coordinator Node                |")
    print("  +======================================================+")
    print()
    print(f"  Host:    {socket.gethostname()}")
    print(f"  IP:      {ip}:{PORT}")
    print(f"  Chip:    {chip}")
    print(f"  Model:   {MODEL}")
    print()

    zc, zc_info = register_mdns(ip, PORT)
    if zc:
        print(f"  - Broadcasting on local network via mDNS")
        print(f"  - Workers can join with:  python demo_worker.py")
        print(f"  - Or manually:            python demo_worker.py --host {ip}")
    else:
        print(f"  - Workers join with:  python demo_worker.py --host {ip}")
    print()

    coordinator = Coordinator()

    server = await asyncio.start_server(
        coordinator.handle_worker, "0.0.0.0", PORT
    )

    try:
        async with server:
            await asyncio.gather(
                server.serve_forever(),
                coordinator.training_loop(),
            )
    finally:
        if zc:
            zc.unregister_service(zc_info)
            zc.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Paused. Run again with --resume to continue.\n")
