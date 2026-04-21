"""Sleep Swarm scheduler — automatic overnight distributed training.

Your Mac auto-joins training sessions while you sleep. Set a training window
(e.g., 11pm-7am), and AirTrain forms overnight swarms by matching people
across time zones. The model trains 24/7 by chasing nighttime around the globe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from airtrain.config import SleepConfig

logger = logging.getLogger("airtrain.sleep")


def _get_local_timezone() -> str:
    """Auto-detect the system's local timezone."""
    try:
        return str(datetime.now().astimezone().tzinfo)
    except Exception:
        return "UTC"


def _get_battery_percent() -> int:
    """Get current battery percentage on macOS. Returns 100 if on AC/unknown."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "%" in line:
                # Parse "99%; charging" or "45%; discharging"
                pct = line.split("%")[0].strip().split()[-1]
                return int(pct)
    except Exception:
        pass
    return 100


def _is_lid_open() -> bool:
    """Check if MacBook lid is open. Returns True if desktop or unknown."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "AppleClamshellState" in result.stdout:
            return '"AppleClamshellState" = No' in result.stdout
    except Exception:
        pass
    return True


def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' into (hour, minute)."""
    parts = time_str.strip().split(":")
    return int(parts[0]), int(parts[1])


def is_within_window(config: SleepConfig) -> bool:
    """Check if the current local time is within the sleep training window."""
    tz = ZoneInfo(config.timezone) if config.timezone else None
    now = datetime.now(tz)
    current_minutes = now.hour * 60 + now.minute

    start_h, start_m = _parse_time(config.window_start)
    end_h, end_m = _parse_time(config.window_end)
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    if start_minutes <= end_minutes:
        # Same-day window (e.g., 09:00-17:00)
        return start_minutes <= current_minutes < end_minutes
    else:
        # Overnight window (e.g., 23:00-07:00)
        return current_minutes >= start_minutes or current_minutes < end_minutes


def minutes_until_window(config: SleepConfig) -> float:
    """Minutes until the next training window opens."""
    tz = ZoneInfo(config.timezone) if config.timezone else None
    now = datetime.now(tz)
    current_minutes = now.hour * 60 + now.minute

    start_h, start_m = _parse_time(config.window_start)
    start_minutes = start_h * 60 + start_m

    if current_minutes < start_minutes:
        return start_minutes - current_minutes
    else:
        return (24 * 60 - current_minutes) + start_minutes


class SleepSwarmSession:
    """Represents a sleep swarm session fetched from the relay server."""

    def __init__(self, data: dict):
        self.id: str = data.get("id", "")
        self.model_name: str = data.get("model_name", "")
        self.checkpoint_url: str = data.get("checkpoint_url", "")
        self.description: str = data.get("description", "")
        self.active_sleepers: int = data.get("active_sleepers", 0)
        self.total_compute_hours: float = data.get("total_compute_hours", 0.0)
        self.coverage_hours: list[bool] = data.get("coverage_hours", [False] * 24)
        self.connect_address: str = data.get("connect_address", "")


class SleepScheduler:
    """Manages the sleep training window and session lifecycle.

    The scheduler:
    1. Waits until the training window opens
    2. Queries the relay server for active sleep swarm sessions
    3. Joins the best matching session as a worker
    4. Monitors battery, lid state, and window boundaries
    5. Gracefully disconnects when the window closes
    """

    def __init__(self, config: SleepConfig):
        self.config = config
        if not config.timezone:
            self.config.timezone = _get_local_timezone()
        self.session: Optional[SleepSwarmSession] = None
        self.start_time: Optional[float] = None
        self.compute_hours: float = 0.0
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def find_session(self) -> Optional[SleepSwarmSession]:
        """Query the relay server for available sleep swarm sessions."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {}
                if self.config.prefer_model != "any":
                    params["model"] = self.config.prefer_model

                resp = await client.get(
                    f"{self.config.relay_url}/sleep/sessions",
                    params=params,
                )
                if resp.status_code == 200:
                    sessions = resp.json()
                    if sessions:
                        # Pick the session with the most coverage gaps
                        # (our timezone can help fill those gaps)
                        best = max(
                            sessions,
                            key=lambda s: s.get("coverage_hours", []).count(False),
                        )
                        return SleepSwarmSession(best)
        except Exception as e:
            logger.warning(f"Failed to query relay server: {e}")
        return None

    async def register_availability(self) -> None:
        """Register this Mac's sleep window with the relay server."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"{self.config.relay_url}/sleep/register",
                    json={
                        "timezone": self.config.timezone,
                        "window_start": self.config.window_start,
                        "window_end": self.config.window_end,
                        "prefer_model": self.config.prefer_model,
                        "battery": _get_battery_percent(),
                    },
                )
        except Exception as e:
            logger.debug(f"Failed to register availability: {e}")

    async def _dream_while_idle(self) -> None:
        """Run dream sessions when no training sessions are available.

        'If you can't train, dream.' Generates synthetic data from the
        last known model checkpoint and caches it for future sessions.
        """
        try:
            from airtrain.config import DreamConfig
            from airtrain.engine.dream import DreamSession

            dream_config = DreamConfig(samples_per_session=200)
            session = DreamSession(dream_config, peer_id=self.config.timezone)
            stats = session.run(num_samples=200)
            logger.info(
                f"Dream session: {stats['kept']} samples cached "
                f"(avg quality {stats['avg_quality']})"
            )
        except Exception as e:
            logger.debug(f"Dream session failed: {e}")

        # Wait before retrying session search
        await asyncio.sleep(300)

    def _should_stop(self) -> tuple[bool, str]:
        """Check if we should stop training. Returns (should_stop, reason)."""
        if not is_within_window(self.config):
            return True, "training window closed"

        battery = _get_battery_percent()
        if battery < self.config.min_battery:
            return True, f"battery low ({battery}%)"

        if not _is_lid_open():
            return True, "lid closed"

        if self.start_time:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.config.max_hours:
                return True, f"max hours reached ({self.config.max_hours}h)"

        return False, ""

    async def _training_loop(self, session: SleepSwarmSession) -> None:
        """Run the training worker for the matched session."""
        from airtrain.config import NetworkConfig

        logger.info(f"Joining sleep swarm: {session.model_name}")
        logger.info(f"  Session: {session.id}")
        logger.info(f"  Connect: {session.connect_address}")

        # Import here to avoid circular imports
        try:
            from airtrain.engine.worker import run_worker
        except ImportError:
            logger.error("Worker engine not available")
            return

        network_config = NetworkConfig()
        address = session.connect_address or "auto"

        try:
            await run_worker(address, network_config)
        except Exception as e:
            logger.error(f"Training error: {e}")

    async def run(self) -> None:
        """Main scheduler loop.

        Waits for the training window, finds a session, trains,
        and repeats nightly.
        """
        self._running = True
        logger.info("Sleep Swarm scheduler started")
        logger.info(f"  Window: {self.config.window_start} - {self.config.window_end}")
        logger.info(f"  Timezone: {self.config.timezone}")
        logger.info(f"  Prefer: {self.config.prefer_model}")
        logger.info(f"  Max hours/night: {self.config.max_hours}")
        logger.info(f"  Min battery: {self.config.min_battery}%")

        while self._running:
            # Wait for the training window to open
            if not is_within_window(self.config):
                wait_mins = minutes_until_window(self.config)
                logger.info(
                    f"Outside training window. Next window in {wait_mins:.0f} minutes."
                )
                # Check every minute
                await asyncio.sleep(min(wait_mins * 60, 60))
                continue

            # Window is open — find a session
            logger.info("Training window is open. Searching for sleep swarm sessions...")
            await self.register_availability()

            session = await self.find_session()
            if not session:
                logger.info("No sessions found. Dreaming instead...")
                await self._dream_while_idle()
                continue

            self.session = session
            self.start_time = time.time()
            logger.info(f"Matched session: {session.model_name} ({session.id})")

            # Start training in a task so we can monitor stop conditions
            self._worker_task = asyncio.create_task(self._training_loop(session))

            # Monitor stop conditions while training
            while not self._worker_task.done():
                should_stop, reason = self._should_stop()
                if should_stop:
                    logger.info(f"Stopping training: {reason}")
                    self._worker_task.cancel()
                    try:
                        await self._worker_task
                    except asyncio.CancelledError:
                        pass
                    break
                await asyncio.sleep(30)  # check every 30 seconds

            # Record compute hours
            if self.start_time:
                self.compute_hours += (time.time() - self.start_time) / 3600
                logger.info(
                    f"Session complete. Contributed {self.compute_hours:.2f} hours tonight."
                )
                self.start_time = None

            # If window is still open, try to find another session
            if is_within_window(self.config):
                logger.info("Window still open. Looking for another session...")
                await asyncio.sleep(10)
            else:
                # Wait for next night
                logger.info("Training window closed. Sleeping until tomorrow.")
                await asyncio.sleep(60)

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        logger.info(
            f"Sleep scheduler stopped. Total compute: {self.compute_hours:.2f} hours"
        )


async def run_sleep_scheduler(config: SleepConfig) -> None:
    """Entry point for the sleep swarm scheduler."""
    scheduler = SleepScheduler(config)
    try:
        await scheduler.run()
    except KeyboardInterrupt:
        scheduler.stop()
    except asyncio.CancelledError:
        scheduler.stop()
