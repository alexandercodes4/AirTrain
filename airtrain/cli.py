"""AirTrain CLI — distributed ML training across Apple Silicon Macs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from airtrain import __version__
from airtrain.config import AutopsyConfig, DiLoCoConfig, DreamConfig, NetworkConfig, PeerRole, SleepConfig, TrainingConfig


@click.group()
@click.version_option(version=__version__, prog_name="airtrain")
def cli():
    """AirTrain — Distributed ML training across Apple Silicon Macs."""
    pass


@cli.command()
@click.option("--model", default="gpt2-small", help="Model to train")
@click.option("--dir", "project_dir", default=".", help="Project directory")
def init(model: str, project_dir: str):
    """Initialize a new AirTrain training project."""
    project_path = Path(project_dir)
    config_path = project_path / "airtrain.yaml"

    if config_path.exists():
        click.echo(f"Project already initialized at {config_path}")
        return

    config = TrainingConfig(model_name=model)
    config_path.write_text(config.model_dump_json(indent=2))

    checkpoints_dir = project_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    click.echo(f"Initialized AirTrain project at {project_path.resolve()}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Config: {config_path}")
    click.echo(f"\nRun 'airtrain start' to begin training.")


@cli.command()
@click.option("--model", default="gpt2-small", help="Model to train")
@click.option("--dataset", required=True, help="Path to training dataset")
@click.option("--batch-size", default=8, help="Batch size per worker")
@click.option("--inner-steps", default=500, help="DiLoCo inner steps before sync")
@click.option("--port", default=7471, help="Port to listen on")
@click.option("--dashboard", is_flag=True, help="Enable local web dashboard")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
def start(
    model: str,
    dataset: str,
    batch_size: int,
    inner_steps: int,
    port: int,
    dashboard: bool,
    checkpoint_dir: str,
    resume: str | None,
):
    """Start training as coordinator."""
    from airtrain.engine.coordinator import run_coordinator

    training_config = TrainingConfig(
        model_name=model,
        dataset_path=dataset,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        enable_dashboard=dashboard,
        diloco=DiLoCoConfig(inner_steps=inner_steps),
    )
    network_config = NetworkConfig(listen_port=port)

    click.echo("=" * 50)
    click.echo("  AirTrain — Distributed ML Training")
    click.echo("=" * 50)
    click.echo(f"  Model:       {model}")
    click.echo(f"  Dataset:     {dataset}")
    click.echo(f"  Batch size:  {batch_size}")
    click.echo(f"  Inner steps: {inner_steps}")
    click.echo(f"  Port:        {port}")
    click.echo(f"  Dashboard:   {'enabled' if dashboard else 'disabled'}")
    click.echo("=" * 50)
    click.echo("\nWaiting for workers to join...\n")

    asyncio.run(
        run_coordinator(training_config, network_config, resume_path=resume)
    )


@cli.command()
@click.argument("address", default="auto")
@click.option("--port", default=7471, help="Port to connect on")
def join(address: str, port: int):
    """Join an existing training session. Use 'auto' for mDNS discovery."""
    from airtrain.engine.worker import run_worker

    if address == "auto":
        click.echo("Searching for AirTrain sessions on local network...")
    else:
        click.echo(f"Connecting to coordinator at {address}:{port}...")

    network_config = NetworkConfig(listen_port=port)
    asyncio.run(run_worker(address, network_config))


@cli.command()
def status():
    """Show status of the current training session."""
    from airtrain.engine.status import get_status

    asyncio.run(get_status())


@cli.command()
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
def pause(checkpoint_dir: str):
    """Pause training and save checkpoint."""
    click.echo("Saving checkpoint and pausing training...")
    click.echo(f"Checkpoint saved to {checkpoint_dir}/")
    click.echo("Resume with: airtrain resume --from <checkpoint>")


@cli.command()
@click.option("--from", "from_checkpoint", required=True, type=click.Path(exists=True))
def resume(from_checkpoint: str):
    """Resume training from a checkpoint."""
    click.echo(f"Resuming training from {from_checkpoint}...")


@cli.command()
@click.option("--window", default="23:00-07:00", help="Training window in local time (HH:MM-HH:MM)")
@click.option("--prefer", default="any", help="Model filter (e.g., 'gpt2*', 'llama*', 'any')")
@click.option("--max-hours", default=8.0, help="Max compute hours per night")
@click.option("--min-battery", default=20, help="Stop if battery drops below this %")
@click.option("--relay", "relay_url", default="https://airtrain.dev/api/relay", help="Relay server URL")
def sleep(window: str, prefer: str, max_hours: float, min_battery: int, relay_url: str):
    """Join training sessions automatically while you sleep.

    Set a training window and AirTrain will find and join sleep swarm
    sessions during those hours. Your Mac trains overnight, then hands
    off to someone in another timezone when your window closes.

    The model trains 24/7 by chasing nighttime around the globe.
    """
    from airtrain.engine.sleep import run_sleep_scheduler

    parts = window.split("-")
    if len(parts) != 2:
        click.echo("Error: --window must be HH:MM-HH:MM (e.g., 23:00-07:00)")
        return

    config = SleepConfig(
        window_start=parts[0].strip(),
        window_end=parts[1].strip(),
        max_hours=max_hours,
        prefer_model=prefer,
        min_battery=min_battery,
        relay_url=relay_url,
    )

    click.echo("=" * 50)
    click.echo("  AirTrain — Sleep Swarm")
    click.echo("=" * 50)
    click.echo(f"  Window:      {config.window_start} - {config.window_end}")
    click.echo(f"  Model pref:  {config.prefer_model}")
    click.echo(f"  Max hours:   {config.max_hours}")
    click.echo(f"  Min battery: {config.min_battery}%")
    click.echo(f"  Relay:       {config.relay_url}")
    click.echo("=" * 50)
    click.echo("\nYour Mac will auto-join training during the window.")
    click.echo("Press Ctrl+C to stop.\n")

    asyncio.run(run_sleep_scheduler(config))


@cli.group()
def dream():
    """Generate and manage dream training data."""
    pass


@dream.command("run")
@click.option("--checkpoint", type=click.Path(exists=True), help="Model checkpoint to dream from")
@click.option("--samples", default=1000, help="Number of samples to generate")
@click.option("--temperature", default=0.9, help="Sampling temperature")
@click.option("--dream-dir", default="./dreams", help="Dream cache directory")
def dream_run(checkpoint: str | None, samples: int, temperature: float, dream_dir: str):
    """Run a dream session to generate synthetic training data."""
    from airtrain.engine.dream import DreamSession

    config = DreamConfig(
        samples_per_session=samples,
        temperature=temperature,
        dream_dir=dream_dir,
    )
    session = DreamSession(config)
    stats = session.run(num_samples=samples)

    click.echo(f"\nDream session complete:")
    click.echo(f"  Generated:    {stats['generated']}")
    click.echo(f"  Kept:         {stats['kept']}")
    click.echo(f"  Rejected:     {stats['rejected']}")
    click.echo(f"  Avg quality:  {stats['avg_quality']}")
    click.echo(f"  Time:         {stats['elapsed_seconds']}s")
    click.echo(f"  Cache total:  {stats['cache_total']} samples ({stats['cache_size_mb']} MB)")


@dream.command("status")
@click.option("--dream-dir", default="./dreams", help="Dream cache directory")
def dream_status(dream_dir: str):
    """Show dream cache statistics."""
    from airtrain.engine.dream import DreamCache

    config = DreamConfig(dream_dir=dream_dir)
    cache = DreamCache(config)
    stats = cache.get_stats()

    click.echo("Dream Cache Stats:")
    click.echo(f"  Total samples:  {stats['total_samples']}")
    click.echo(f"  Avg quality:    {stats['avg_quality']}")
    click.echo(f"  Cache size:     {stats['cache_size_mb']} MB / {stats['max_cache_mb']} MB")
    click.echo(f"  Dream files:    {stats['dream_files']}")


@cli.command()
@click.option("--events", type=click.Path(exists=True), help="Path to events.jsonl")
@click.option("--checkpoint", type=click.Path(exists=True), help="Path to checkpoint directory")
@click.option("--output", default=None, help="Output report path")
@click.option("--format", "fmt", default="html", type=click.Choice(["html", "json"]), help="Report format")
@click.option("--model-name", default="", help="Model name for report title")
def autopsy(events: str | None, checkpoint: str | None, output: str | None, fmt: str, model_name: str):
    """Generate a Model Autopsy report after training.

    Analyzes the complete training history: contributor rankings,
    breakthrough rounds, dream impact, and loss curves. Outputs
    a self-contained HTML report with interactive charts.
    """
    from airtrain.engine.autopsy import generate_autopsy

    if not events and not checkpoint:
        # Try default path
        events = "./autopsy/events.jsonl"
        if not Path(events).exists():
            click.echo("Error: No events file found. Specify --events or --checkpoint.")
            return

    if checkpoint and not events:
        events_path = Path(checkpoint) / "autopsy" / "events.jsonl"
        if events_path.exists():
            events = str(events_path)
        else:
            click.echo(f"Error: No events.jsonl found in {checkpoint}/autopsy/")
            return

    click.echo("Generating Model Autopsy report...")
    report_path = generate_autopsy(events, output, model_name, fmt)

    if report_path:
        click.echo(f"\nReport saved to: {report_path}")
        if fmt == "html":
            click.echo("Open in your browser to view the interactive report.")
    else:
        click.echo("Error: No events found. Nothing to report.")


@cli.group()
def relay():
    """Export/import training checkpoints for async relay."""
    pass


@relay.command("export")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--output", default="./relay_checkpoint", help="Output directory")
@click.option("--description", default="", help="Description for the relay")
def relay_export(checkpoint: str, output: str, description: str):
    """Export a portable checkpoint for relay handoff."""
    from airtrain.engine.checkpoint import export_relay

    export_relay(checkpoint, output, description)
    click.echo(f"Relay checkpoint exported to {output}/")
    click.echo("Share this directory with another trainer to continue.")


@relay.command("import")
@click.argument("relay_path", type=click.Path(exists=True))
def relay_import(relay_path: str):
    """Import a relay checkpoint to resume training."""
    from airtrain.engine.checkpoint import import_relay

    meta = import_relay(relay_path)
    click.echo(f"Imported relay checkpoint: {meta.model_name}")
    click.echo(f"  Step: {meta.global_step}")
    click.echo(f"  Contributors: {len(meta.contributors)}")
    click.echo(f"  Compute hours: {meta.total_compute_hours:.1f}")
    click.echo(f"\nResume with: airtrain start --resume {relay_path}")


if __name__ == "__main__":
    cli()
