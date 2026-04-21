"""Model Autopsy — automated post-training analysis and report generation.

After training completes, generates a detailed report showing the model's
entire training life story: which peers contributed most, which rounds
caused the biggest loss drops, dream impact analysis, and a full timeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from airtrain.config import AutopsyConfig

logger = logging.getLogger("airtrain.autopsy")


@dataclass
class TrainingEvent:
    """A single recorded event during training."""

    event_type: str  # sync, join, leave, checkpoint, dream_session
    timestamp: float = field(default_factory=time.time)
    global_step: int = 0
    loss: float = 0.0
    peer_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "ts": self.timestamp,
            "step": self.global_step,
            "loss": self.loss,
            "peer": self.peer_id,
            "meta": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingEvent:
        return cls(
            event_type=data["event"],
            timestamp=data.get("ts", 0.0),
            global_step=data.get("step", 0),
            loss=data.get("loss", 0.0),
            peer_id=data.get("peer", ""),
            metadata=data.get("meta", {}),
        )


class AutopsyRecorder:
    """Collects training events during a session.

    Attach this to the coordinator or worker to automatically
    record events as they happen. Events are periodically
    flushed to disk as JSONL.
    """

    def __init__(self, config: AutopsyConfig):
        self.config = config
        self.events: list[TrainingEvent] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self.output_dir / "events.jsonl"
        self._start_time = time.time()
        self._model_name = ""

    def set_model_name(self, name: str) -> None:
        self._model_name = name

    def record_sync(
        self, step: int, loss: float, peers: list[str], gradients_received: int
    ) -> None:
        self._add(TrainingEvent(
            event_type="sync",
            global_step=step,
            loss=loss,
            metadata={
                "peers": peers,
                "gradients": gradients_received,
                "peer_count": len(peers),
            },
        ))

    def record_peer_join(self, peer_id: str, chip: str = "", memory_gb: float = 0.0) -> None:
        self._add(TrainingEvent(
            event_type="join",
            peer_id=peer_id,
            metadata={"chip": chip, "memory_gb": memory_gb},
        ))

    def record_peer_leave(self, peer_id: str, compute_hours: float = 0.0) -> None:
        self._add(TrainingEvent(
            event_type="leave",
            peer_id=peer_id,
            metadata={"compute_hours": compute_hours},
        ))

    def record_checkpoint(self, step: int, loss: float, path: str = "") -> None:
        self._add(TrainingEvent(
            event_type="checkpoint",
            global_step=step,
            loss=loss,
            metadata={"path": path},
        ))

    def record_dream_session(
        self, peer_id: str, generated: int, kept: int, avg_quality: float
    ) -> None:
        self._add(TrainingEvent(
            event_type="dream_session",
            peer_id=peer_id,
            metadata={
                "generated": generated,
                "kept": kept,
                "avg_quality": avg_quality,
            },
        ))

    def _add(self, event: TrainingEvent) -> None:
        self.events.append(event)
        # Flush every 50 events
        if len(self.events) % 50 == 0:
            self.flush()

    def flush(self) -> None:
        """Write pending events to disk."""
        if not self.events:
            return
        with open(self._events_file, "a") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + "\n")
        count = len(self.events)
        self.events = []
        logger.debug(f"Flushed {count} events to {self._events_file}")

    def finalize(self) -> None:
        """Flush remaining events and log summary."""
        self.flush()
        logger.info(f"Autopsy events saved to {self._events_file}")

    @classmethod
    def load_events(cls, path: str | Path) -> list[TrainingEvent]:
        """Load events from a JSONL file."""
        events = []
        path = Path(path)
        if not path.exists():
            return events
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(TrainingEvent.from_dict(json.loads(line)))
        return events


class AutopsyAnalyzer:
    """Computes insights from recorded training events."""

    def __init__(self, events: list[TrainingEvent]):
        self.events = sorted(events, key=lambda e: e.timestamp)
        self._syncs = [e for e in self.events if e.event_type == "sync"]
        self._joins = [e for e in self.events if e.event_type == "join"]
        self._leaves = [e for e in self.events if e.event_type == "leave"]
        self._checkpoints = [e for e in self.events if e.event_type == "checkpoint"]
        self._dreams = [e for e in self.events if e.event_type == "dream_session"]

    def training_summary(self) -> dict:
        """Overall training summary."""
        if not self.events:
            return {"total_steps": 0, "total_time_hours": 0}

        first_ts = self.events[0].timestamp
        last_ts = self.events[-1].timestamp
        total_hours = (last_ts - first_ts) / 3600

        all_peers = set()
        for e in self.events:
            if e.peer_id:
                all_peers.add(e.peer_id)
            if "peers" in e.metadata:
                all_peers.update(e.metadata["peers"])

        final_loss = self._syncs[-1].loss if self._syncs else 0.0
        initial_loss = self._syncs[0].loss if self._syncs else 0.0
        max_step = max((e.global_step for e in self.events), default=0)

        total_compute = sum(
            e.metadata.get("compute_hours", 0.0)
            for e in self._leaves
        )

        return {
            "total_steps": max_step,
            "total_time_hours": round(total_hours, 2),
            "total_compute_hours": round(total_compute, 2),
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_reduction": round(initial_loss - final_loss, 4),
            "total_syncs": len(self._syncs),
            "total_peers": len(all_peers),
            "total_checkpoints": len(self._checkpoints),
            "total_dream_sessions": len(self._dreams),
            "start_time": datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat(),
        }

    def top_contributors(self) -> list[dict]:
        """Rank peers by contribution metrics."""
        peers: dict[str, dict] = {}

        for e in self._joins:
            pid = e.peer_id
            if pid not in peers:
                peers[pid] = {
                    "peer_id": pid,
                    "chip": e.metadata.get("chip", "unknown"),
                    "memory_gb": e.metadata.get("memory_gb", 0),
                    "compute_hours": 0.0,
                    "syncs_participated": 0,
                    "dreams_generated": 0,
                    "join_time": e.timestamp,
                }

        for e in self._leaves:
            if e.peer_id in peers:
                peers[e.peer_id]["compute_hours"] += e.metadata.get("compute_hours", 0.0)

        for e in self._syncs:
            for pid in e.metadata.get("peers", []):
                if pid in peers:
                    peers[pid]["syncs_participated"] += 1

        for e in self._dreams:
            if e.peer_id in peers:
                peers[e.peer_id]["dreams_generated"] += e.metadata.get("kept", 0)

        ranked = sorted(peers.values(), key=lambda p: p["compute_hours"], reverse=True)
        for i, p in enumerate(ranked):
            p["rank"] = i + 1
            p["compute_hours"] = round(p["compute_hours"], 2)
        return ranked

    def loss_milestones(self, top_n: int = 5) -> list[dict]:
        """Find the rounds with the biggest loss drops (breakthroughs)."""
        if len(self._syncs) < 2:
            return []

        drops = []
        for i in range(1, len(self._syncs)):
            prev = self._syncs[i - 1]
            curr = self._syncs[i]
            drop = prev.loss - curr.loss
            if drop > 0:
                drops.append({
                    "step": curr.global_step,
                    "loss_before": round(prev.loss, 4),
                    "loss_after": round(curr.loss, 4),
                    "drop": round(drop, 4),
                    "drop_pct": round(drop / max(prev.loss, 1e-8) * 100, 1),
                    "peers": curr.metadata.get("peers", []),
                    "peer_count": curr.metadata.get("peer_count", 0),
                })

        drops.sort(key=lambda d: d["drop"], reverse=True)
        return drops[:top_n]

    def dream_impact(self) -> dict:
        """Analyze the impact of dream training on loss improvement."""
        if not self._dreams:
            return {"dreams_used": False}

        total_generated = sum(d.metadata.get("generated", 0) for d in self._dreams)
        total_kept = sum(d.metadata.get("kept", 0) for d in self._dreams)
        avg_quality = (
            sum(d.metadata.get("avg_quality", 0) for d in self._dreams) / len(self._dreams)
        )

        return {
            "dreams_used": True,
            "total_sessions": len(self._dreams),
            "total_generated": total_generated,
            "total_kept": total_kept,
            "keep_rate": round(total_kept / max(total_generated, 1) * 100, 1),
            "avg_quality": round(avg_quality, 3),
            "dreamers": list(set(d.peer_id for d in self._dreams)),
        }

    def peer_timeline(self) -> list[dict]:
        """Generate timeline of when each peer was active."""
        peers: dict[str, dict] = {}

        for e in self._joins:
            peers[e.peer_id] = {
                "peer_id": e.peer_id,
                "chip": e.metadata.get("chip", "unknown"),
                "joined_at": e.timestamp,
                "left_at": None,
            }

        for e in self._leaves:
            if e.peer_id in peers:
                peers[e.peer_id]["left_at"] = e.timestamp

        # For peers that never left, use the last event timestamp
        last_ts = self.events[-1].timestamp if self.events else time.time()
        for p in peers.values():
            if p["left_at"] is None:
                p["left_at"] = last_ts

        return sorted(peers.values(), key=lambda p: p["joined_at"])

    def loss_curve(self) -> list[dict]:
        """Extract the loss curve data points."""
        return [
            {"step": e.global_step, "loss": round(e.loss, 4), "ts": e.timestamp}
            for e in self._syncs
        ]


class AutopsyReport:
    """Generates human-readable autopsy reports."""

    def __init__(self, analyzer: AutopsyAnalyzer, model_name: str = ""):
        self.analyzer = analyzer
        self.model_name = model_name

    def generate_json(self) -> dict:
        """Generate a machine-readable JSON report."""
        return {
            "model_name": self.model_name,
            "summary": self.analyzer.training_summary(),
            "contributors": self.analyzer.top_contributors(),
            "breakthroughs": self.analyzer.loss_milestones(),
            "dream_impact": self.analyzer.dream_impact(),
            "loss_curve": self.analyzer.loss_curve(),
            "peer_timeline": self.analyzer.peer_timeline(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate_html(self) -> str:
        """Generate a self-contained HTML report with charts."""
        data = self.generate_json()
        summary = data["summary"]
        contributors = data["contributors"]
        breakthroughs = data["breakthroughs"]
        dream = data["dream_impact"]
        loss_curve = data["loss_curve"]

        loss_labels = json.dumps([p["step"] for p in loss_curve])
        loss_values = json.dumps([p["loss"] for p in loss_curve])

        contributors_rows = ""
        for c in contributors:
            contributors_rows += f"""
            <tr>
              <td>{c['rank']}</td>
              <td>{c['peer_id']}</td>
              <td>{c['chip']}</td>
              <td>{c['memory_gb']} GB</td>
              <td>{c['compute_hours']} hrs</td>
              <td>{c['syncs_participated']}</td>
              <td>{c['dreams_generated']}</td>
            </tr>"""

        breakthroughs_rows = ""
        for b in breakthroughs:
            breakthroughs_rows += f"""
            <tr>
              <td>Step {b['step']}</td>
              <td>{b['loss_before']}</td>
              <td>{b['loss_after']}</td>
              <td>-{b['drop']} ({b['drop_pct']}%)</td>
              <td>{b['peer_count']} peers</td>
            </tr>"""

        dream_section = ""
        if dream.get("dreams_used"):
            dream_section = f"""
        <div class="card">
          <h2>Dream Training Impact</h2>
          <div class="stats-grid">
            <div class="stat"><span class="stat-val">{dream['total_sessions']}</span><span class="stat-label">Dream Sessions</span></div>
            <div class="stat"><span class="stat-val">{dream['total_kept']}</span><span class="stat-label">Samples Kept</span></div>
            <div class="stat"><span class="stat-val">{dream['keep_rate']}%</span><span class="stat-label">Keep Rate</span></div>
            <div class="stat"><span class="stat-val">{dream['avg_quality']}</span><span class="stat-label">Avg Quality</span></div>
          </div>
          <p style="margin-top:16px;color:#a1a1b0;">Dreamers: {', '.join(dream.get('dreamers', []))}</p>
        </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Autopsy — {self.model_name or 'AirTrain'}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    :root {{
      --bg: #08080a;
      --surface: #0f0f13;
      --elevated: #16161c;
      --bright: #e4e4ec;
      --mid: #a1a1b0;
      --dim: #6b6b78;
      --subtle: #3a3a44;
    }}
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family: 'Inter', -apple-system, system-ui, sans-serif;
      background: var(--bg);
      color: var(--mid);
      line-height: 1.6;
      padding: 32px;
      max-width: 960px;
      margin: 0 auto;
    }}
    h1 {{ color: var(--bright); font-size: 1.75rem; font-weight: 700; margin-bottom: 8px; }}
    h2 {{ color: var(--bright); font-size: 1.125rem; font-weight: 600; margin-bottom: 16px; }}
    .subtitle {{ color: var(--dim); font-size: 0.875rem; margin-bottom: 32px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--subtle);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 16px;
    }}
    .stat {{ text-align: center; }}
    .stat-val {{ display: block; font-size: 1.5rem; font-weight: 700; color: var(--bright); }}
    .stat-label {{ font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; padding: 8px 12px; font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--subtle); }}
    td {{ padding: 8px 12px; font-size: 0.8125rem; border-bottom: 1px solid rgba(255,255,255,0.04); }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .chart-wrap {{ height: 300px; margin-top: 16px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.6875rem; background: rgba(255,255,255,0.06); color: var(--bright); }}
  </style>
</head>
<body>
  <h1>Model Autopsy</h1>
  <p class="subtitle">{self.model_name or 'AirTrain Model'} &mdash; Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

  <div class="card">
    <h2>Training Summary</h2>
    <div class="stats-grid">
      <div class="stat"><span class="stat-val">{summary['total_steps']:,}</span><span class="stat-label">Total Steps</span></div>
      <div class="stat"><span class="stat-val">{summary['total_time_hours']}</span><span class="stat-label">Hours Elapsed</span></div>
      <div class="stat"><span class="stat-val">{summary['total_compute_hours']}</span><span class="stat-label">Compute Hours</span></div>
      <div class="stat"><span class="stat-val">{summary['total_peers']}</span><span class="stat-label">Contributors</span></div>
      <div class="stat"><span class="stat-val">{summary['initial_loss']}</span><span class="stat-label">Initial Loss</span></div>
      <div class="stat"><span class="stat-val">{summary['final_loss']}</span><span class="stat-label">Final Loss</span></div>
      <div class="stat"><span class="stat-val">{summary['loss_reduction']}</span><span class="stat-label">Loss Reduced</span></div>
      <div class="stat"><span class="stat-val">{summary['total_syncs']}</span><span class="stat-label">Sync Rounds</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Loss Curve</h2>
    <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
  </div>

  <div class="card">
    <h2>Top Contributors</h2>
    <table>
      <thead><tr><th>#</th><th>Peer</th><th>Chip</th><th>Memory</th><th>Compute</th><th>Syncs</th><th>Dreams</th></tr></thead>
      <tbody>{contributors_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Breakthrough Rounds</h2>
    <p style="color:var(--dim);font-size:0.8125rem;margin-bottom:12px;">Sync rounds with the biggest loss drops</p>
    <table>
      <thead><tr><th>Round</th><th>Before</th><th>After</th><th>Drop</th><th>Peers</th></tr></thead>
      <tbody>{breakthroughs_rows}</tbody>
    </table>
  </div>

  {dream_section}

  <p style="text-align:center;color:var(--subtle);font-size:0.75rem;margin-top:40px;">
    Generated by AirTrain Model Autopsy &mdash; airtrain.dev
  </p>

  <script>
    new Chart(document.getElementById('lossChart'), {{
      type: 'line',
      data: {{
        labels: {loss_labels},
        datasets: [{{
          label: 'Training Loss',
          data: {loss_values},
          borderColor: '#e4e4ec',
          backgroundColor: 'rgba(228,228,236,0.05)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ display: true, title: {{ display: true, text: 'Step', color: '#6b6b78' }}, ticks: {{ color: '#6b6b78' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
          y: {{ display: true, title: {{ display: true, text: 'Loss', color: '#6b6b78' }}, ticks: {{ color: '#6b6b78' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        }},
        plugins: {{ legend: {{ display: false }} }},
      }}
    }});
  </script>
</body>
</html>"""


def generate_autopsy(
    events_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = "",
    output_format: str = "html",
) -> str:
    """Generate an autopsy report from an events log.

    Args:
        events_path: Path to events.jsonl file.
        output_path: Where to save the report. If None, auto-generates.
        model_name: Name of the model for the report title.
        output_format: "html" or "json".

    Returns:
        Path to the generated report file.
    """
    events = AutopsyRecorder.load_events(events_path)
    if not events:
        logger.warning(f"No events found in {events_path}")
        return ""

    analyzer = AutopsyAnalyzer(events)
    report = AutopsyReport(analyzer, model_name)

    if output_path is None:
        events_dir = Path(events_path).parent
        ext = "html" if output_format == "html" else "json"
        output_path = events_dir / f"report.{ext}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        content = json.dumps(report.generate_json(), indent=2)
    else:
        content = report.generate_html()

    output_path.write_text(content)
    logger.info(f"Autopsy report saved to {output_path}")
    return str(output_path)
