#!/usr/bin/env python3
"""
AirTrain Demo - Worker (Mac 2)

Run this on the second Mac. It auto-discovers the coordinator on the local
network via mDNS and joins the training swarm.

Setup:
    pip install zeroconf

Run:
    python demo_worker.py              # auto-discover via mDNS
    python demo_worker.py --host <IP>  # connect directly if on different subnet
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import math
import random
import socket
import struct
import sys
import time

try:
    from zeroconf import ServiceBrowser, Zeroconf
    # ServiceListener was removed in zeroconf >= 0.38 — use plain class
    try:
        from zeroconf import ServiceListener as _SL
        _BaseListener = _SL
    except ImportError:
        _BaseListener = object
    HAS_ZEROCONF = True
except ImportError:
    _BaseListener = object
    HAS_ZEROCONF = False

# -- Config --------------------------------------------------------------------
PORT = 7471
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

# -- mDNS Discovery ------------------------------------------------------------

class AirTrainListener(_BaseListener):
    def __init__(self):
        self.found = asyncio.get_event_loop().create_future()

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info and not self.found.done():
            ip = socket.inet_ntoa(info.addresses[0])
            port = info.port
            props = {k.decode(): v.decode() for k, v in info.properties.items()}
            self.found.set_result((ip, port, props))

    def remove_service(self, zc, type_, name): pass
    def update_service(self, zc, type_, name): pass


async def discover_coordinator(timeout=15.0):
    """Browse mDNS for an AirTrain coordinator on the LAN."""
    if not HAS_ZEROCONF:
        return None

    loop = asyncio.get_event_loop()
    zc = Zeroconf()
    listener = AirTrainListener()
    browser = ServiceBrowser(zc, "_airtrain._tcp.local.", listener)

    try:
        result = await asyncio.wait_for(
            asyncio.wrap_future(listener.found) if hasattr(listener.found, '_asyncio_future_blocking')
            else listener.found,
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return None
    finally:
        browser.cancel()
        zc.close()

# -- Worker --------------------------------------------------------------------

class Worker:
    def __init__(self, hostname, chip):
        self.hostname = hostname
        self.chip = chip
        self.loss = None
        self.global_step = 0
        self.round = 0
        self.compute_hours = 0.0
        self.start_time = time.time()

    async def run(self, host, port):
        print(f"  Connecting to coordinator at {host}:{port}...")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10.0
            )
        except (ConnectionRefusedError, asyncio.TimeoutError):
            print(f"\n  [x] Could not connect to {host}:{port}")
            print(f"  Make sure demo_coordinator.py is running on the other Mac.\n")
            return

        # Send handshake
        handshake = json.dumps({
            "type": "handshake",
            "hostname": self.hostname,
            "chip": self.chip,
            "memory_gb": 16,
            "role": "worker",
        }).encode()
        writer.write(handshake)
        await writer.drain()

        # Receive ack
        raw = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        ack = json.loads(raw.decode())
        self.global_step = ack.get("global_step", 0)
        self.loss = ack.get("loss", 4.8)
        self.round = ack.get("round", 0)

        print(f"\n  OK Joined swarm!")
        print(f"  OK Synced to step {self.global_step}, loss {fmt_loss(self.loss)}")
        print(f"\n  Waiting for coordinator to start round...\n")

        try:
            while True:
                # Read next message from coordinator
                header = await asyncio.wait_for(reader.readexactly(4), timeout=120.0)
                length = struct.unpack(">I", header)[0]
                data = await reader.readexactly(length)
                msg = json.loads(data.decode())

                if msg["type"] == "train":
                    await self._do_training_round(msg, writer)

                elif msg["type"] == "weights":
                    # Coordinator sent updated global weights
                    new_loss = msg.get("loss", self.loss)
                    self.global_step = msg.get("global_step", self.global_step)
                    self.round = msg.get("round", self.round)
                    self.loss = new_loss

        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError):
            elapsed = time.time() - self.start_time
            self.compute_hours = elapsed / 3600
            print(f"\n  Connection closed by coordinator.")
            print(f"  Contributed {elapsed / 60:.1f} min of compute ({self.compute_hours:.4f} hrs)")

    async def _do_training_round(self, msg, writer):
        rnd = msg["round"]
        steps = msg["steps"]
        self.round = rnd
        self.global_step = msg.get("global_step", self.global_step)

        t0 = time.time()
        loss_before = self.loss
        local_deltas = []

        print(f"  +-- Round {rnd} starting - training {steps} steps locally...")

        for step in range(steps):
            await asyncio.sleep(0.07)   # simulate compute
            decay = math.exp(-self.global_step / 800)
            noise = random.gauss(0, 0.018)
            delta = 0.011 * decay + noise
            local_deltas.append(delta)
            self.global_step += 1

            if (step + 1) % (steps // 4) == 0:
                local_loss = max(0.8, loss_before - sum(local_deltas) * 0.65)
                pct = int((step + 1) / steps * 20)
                prog = "#" * pct + "." * (20 - pct)
                print(f"  |   [{prog}] step {self.global_step} - local loss {fmt_loss(local_loss)}")

        avg_delta = sum(local_deltas) / len(local_deltas)
        elapsed = time.time() - t0
        tokens_per_sec = (steps * BATCH_SIZE * 512) / elapsed

        # Send pseudo-gradient (weight delta) to coordinator
        pseudo_grad_msg = json.dumps({
            "type": "pseudo_grad",
            "delta": avg_delta,
            "steps": steps,
            "hostname": self.hostname,
        }).encode()
        writer.write(struct.pack(">I", len(pseudo_grad_msg)) + pseudo_grad_msg)
        await writer.drain()

        print(f"  +-- Synced pseudo-gradients - coordinator | {tokens_per_sec:,.0f} tok/s")

        # Now wait for updated weights from coordinator
        header = await asyncio.wait_for(writer._transport._extra.get('reader', None) and
                                         asyncio.Future(), timeout=0.001)

    async def run_with_weight_loop(self, host, port):
        """Main entry: wraps run() with a cleaner message loop."""
        print(f"  Connecting to coordinator at {host}:{port}...")
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10.0
            )
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            reader, writer = None, None
            for attempt in range(1, 6):
                print(f"  Retry {attempt}/5 in 2s...")
                await asyncio.sleep(2)
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port), timeout=5.0
                    )
                    break
                except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
                    pass
            if writer is None:
                print(f"\n  [x] Could not connect to {host}:{port} after 5 attempts.")
                print(f"  Make sure demo_coordinator.py is running on the other Mac.\n")
                return

        # Handshake
        handshake = json.dumps({
            "type": "handshake",
            "hostname": self.hostname,
            "chip": self.chip,
            "memory_gb": 16,
            "role": "worker",
        }).encode()
        writer.write(handshake)
        await writer.drain()

        # Ack (coordinator sends newline-delimited JSON for handshake)
        raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
        ack = json.loads(raw.decode().strip())
        self.global_step = ack.get("global_step", 0)
        self.loss = ack.get("loss", 4.8)

        print(f"\n  OK Joined swarm!")
        print(f"  OK Synced to step {self.global_step}, loss {fmt_loss(self.loss)}")
        print(f"\n  Waiting for coordinator to start next round...\n")

        try:
            while True:
                header = await asyncio.wait_for(reader.readexactly(4), timeout=300.0)
                length = struct.unpack(">I", header)[0]
                data = await reader.readexactly(length)
                msg = json.loads(data.decode())

                if msg["type"] == "train":
                    rnd = msg["round"]
                    steps = msg["steps"]
                    self.global_step = msg.get("global_step", self.global_step)
                    t0 = time.time()
                    loss_before = self.loss
                    local_deltas = []

                    print(f"  +-- Round {rnd} - training {steps} steps locally on {self.chip}...")

                    for step in range(steps):
                        await asyncio.sleep(0.07)
                        decay = math.exp(-self.global_step / 800)
                        noise = random.gauss(0, 0.018)
                        delta = 0.011 * decay + noise
                        local_deltas.append(delta)
                        self.global_step += 1

                        if (step + 1) % max(1, steps // 4) == 0:
                            local_loss = max(0.8, loss_before - sum(local_deltas) * 0.65)
                            pct = int((step + 1) / steps * 20)
                            prog = "#" * pct + "." * (20 - pct)
                            print(f"  |   [{prog}] step {self.global_step} - loss {fmt_loss(local_loss)}")

                    avg_delta = sum(local_deltas) / len(local_deltas)
                    elapsed = time.time() - t0
                    tokens_per_sec = (steps * BATCH_SIZE * 512) / elapsed

                    # Send pseudo-gradient to coordinator
                    pg_msg = json.dumps({
                        "type": "pseudo_grad",
                        "delta": avg_delta,
                        "steps": steps,
                        "hostname": self.hostname,
                    }).encode()
                    writer.write(struct.pack(">I", len(pg_msg)) + pg_msg)
                    await writer.drain()
                    print(f"  +-- Gradients synced - coordinator | {tokens_per_sec:,.0f} tok/s")

                elif msg["type"] == "weights":
                    self.loss = msg.get("loss", self.loss)
                    self.global_step = msg.get("global_step", self.global_step)
                    rnd = msg.get("round", "-")
                    print(f"\n  - Round {rnd} complete - global loss {fmt_loss(self.loss)} "
                          f"{bar(self.loss)} step {self.global_step}\n")
                    print(f"  Waiting for next round...\n")

        except (asyncio.IncompleteReadError, ConnectionResetError):
            elapsed = time.time() - self.start_time
            print(f"\n  Session ended.")
            print(f"  Contributed {elapsed / 60:.1f} min of compute to the swarm. Thanks!\n")
        except asyncio.TimeoutError:
            print(f"\n  Timed out waiting for coordinator.\n")


async def main(host=None):
    chip = get_chip_info()
    hostname = socket.gethostname()

    print()
    print("  +======================================================+")
    print("  |           AirTrain - Worker Node                    |")
    print("  +======================================================+")
    print()
    print(f"  Host:    {hostname}")
    print(f"  IP:      {get_local_ip()}")
    print(f"  Chip:    {chip}")
    print()

    if host:
        print(f"  Connecting directly to {host}:{PORT}...\n")
    else:
        if not HAS_ZEROCONF:
            print("  [x] zeroconf not installed - can't auto-discover.")
            print("  Install it: pip install zeroconf")
            print("  Or run: python demo_worker.py --host <coordinator-ip>\n")
            return

        print("  Searching for AirTrain coordinator on local network...")
        print("  (make sure demo_coordinator.py is running on the other Mac)\n")

        result = await discover_coordinator(timeout=20.0)
        if result is None:
            print("  [x] No coordinator found on this network after 20 seconds.")
            print("  Try: python demo_worker.py --host <coordinator-ip>\n")
            return

        host, found_port, props = result
        model = props.get("model", "unknown model")
        coord_chip = props.get("chip", "unknown chip")
        print(f"  - Found coordinator!")
        print(f"  - Address:  {host}:{found_port}")
        print(f"  - Model:    {model}")
        print(f"  - Host chip: {coord_chip}")
        print()

    worker = Worker(hostname=hostname, chip=chip)
    await worker.run_with_weight_loop(host, PORT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AirTrain Worker Demo")
    parser.add_argument("--host", type=str, default=None,
                        help="Coordinator IP address (skip mDNS discovery)")
    args = parser.parse_args()

    try:
        asyncio.run(main(host=args.host))
    except KeyboardInterrupt:
        print("\n  Disconnected from swarm.\n")
