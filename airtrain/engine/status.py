"""Status display for AirTrain training sessions."""

from __future__ import annotations

import asyncio

import httpx


async def get_status(host: str = "localhost", port: int = 8471):
    """Query the dashboard API for training status."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://{host}:{port}/api/status", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Model:      {data.get('model_name', 'unknown')}")
                print(f"Step:       {data.get('global_step', 0)}")
                print(f"Loss:       {data.get('loss', 0):.4f}")
                print(f"Peers:      {data.get('peer_count', 0)}")
                print(f"Throughput: {data.get('throughput', 0):.0f} tokens/sec")
                print(f"Status:     {data.get('status', 'unknown')}")
            else:
                print(f"Error: {resp.status_code}")
    except httpx.ConnectError:
        print("No active training session found.")
        print("Start training with: airtrain start --dashboard")
    except Exception as e:
        print(f"Error: {e}")
