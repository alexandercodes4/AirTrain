#!/usr/bin/env python3
"""
AirTrain Interactive CLI — A beautiful terminal interface for distributed ML training.
Silver/white themed with ASCII art animations and Ollama integration.

Usage:
    python airtrain_cli.py              # Interactive mode
    python airtrain_cli.py --demo       # Run full demo sequence
    python airtrain_cli.py --ollama     # Connect to local Ollama for dream training
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import signal
import struct
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ─── ANSI Color Palette (Silver/White/Grey Theme) ─────────────────────────────

class C:
    """Color constants — silver/white/grey palette."""
    RST      = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    ITALIC   = "\033[3m"
    ULINE    = "\033[4m"
    BLINK    = "\033[5m"

    # Greys & Whites
    WHITE    = "\033[97m"
    SILVER   = "\033[37m"
    GREY     = "\033[90m"
    LGREY    = "\033[38;5;250m"
    MGREY    = "\033[38;5;245m"
    DGREY    = "\033[38;5;240m"

    # Accent colors (subtle)
    CYAN     = "\033[96m"
    GREEN    = "\033[92m"
    YELLOW   = "\033[93m"
    RED      = "\033[91m"
    BLUE     = "\033[94m"
    MAGENTA  = "\033[95m"

    # Backgrounds
    BG_DARK  = "\033[48;5;233m"
    BG_MED   = "\033[48;5;236m"
    BG_LIGHT = "\033[48;5;238m"

    # 256-color silvers
    S1       = "\033[38;5;255m"  # brightest
    S2       = "\033[38;5;253m"
    S3       = "\033[38;5;251m"
    S4       = "\033[38;5;249m"
    S5       = "\033[38;5;247m"
    S6       = "\033[38;5;245m"
    S7       = "\033[38;5;243m"
    S8       = "\033[38;5;241m"


# ─── ASCII Art ─────────────────────────────────────────────────────────────────

LOGO_FRAMES = [
    r"""
     ___    _        _____           _
    /   \  (_) _ __ |_   _|_ __ __ _(_)_ __
   / /\ /  | || '__|  | | | '__/ _` | | '_ \
  / /_//   | || |     | | | | | (_| | | | | |
 /___,'    |_||_|     |_| |_|  \__,_|_|_| |_|
""",
    r"""
     ___    _        _____           _
    /   \  (_) _ __ |_   _|_ __ __ _(_)_ __
   / /\ /  | || '__|  | | | '__/ _` | | '_ \
  / /_//   | || |     | | | | | (_| | | | | |
 /___,'    |_||_|     |_| |_|  \__,_|_|_| |_|
""",
]

TRAIN_ICON = r"""
       ___
  ____/   \____
 |  ___   ___  |
 | |   | |   | |
 | |___| |___| |
 |_____________|
    O       O
"""

LOGO_SMALL = f"""{C.S1}     ___   {C.S3} _        {C.S5}_____          {C.S6} _
{C.S1}    /   \\  {C.S3}(_) _ __ {C.S5}|_   _|_ __ __ _{C.S6}(_)_ __
{C.S1}   / /\\ /  {C.S3}| || '__|{C.S5}  | | | '__/ _` {C.S6}| | '_ \\
{C.S1}  / /_//   {C.S3}| || |   {C.S5}  | | | | | (_| {C.S6}| | | | |
{C.S1} /___,'    {C.S3}|_||_|   {C.S5}  |_| |_|  \\__,_{C.S6}|_|_| |_|{C.RST}"""

# Gradient logo with shimmer effect
def render_gradient_logo(frame: int = 0) -> str:
    """Render the logo with a moving silver gradient shimmer."""
    raw = r"""     ___    _        _____           _
    /   \  (_) _ __  |_   _|_ __ __ _(_)_ __
   / /\ /  | || '__|   | | | '__/ _` | | '_ \
  / /_//   | || |      | | | | | (_| | | | | |
 /___,'    |_||_|      |_| |_|  \__,_|_|_| |_|"""

    colors = [C.S8, C.S7, C.S6, C.S5, C.S4, C.S3, C.S2, C.S1, C.S2, C.S3, C.S4, C.S5, C.S6, C.S7, C.S8]
    lines = raw.split("\n")
    result = []
    for row_idx, line in enumerate(lines):
        colored = ""
        for col_idx, ch in enumerate(line):
            ci = (col_idx + frame * 2 + row_idx) % len(colors)
            colored += colors[ci] + ch
        result.append(colored + C.RST)
    return "\n".join(result)


# ─── Terminal Utilities ────────────────────────────────────────────────────────

def term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def term_height() -> int:
    return shutil.get_terminal_size((80, 24)).lines

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def move_to(row: int, col: int):
    sys.stdout.write(f"\033[{row};{col}H")
    sys.stdout.flush()

def center(text: str, width: int = 0) -> str:
    w = width or term_width()
    lines = text.split("\n")
    return "\n".join(line.center(w) for line in lines)

def box(content: str, title: str = "", width: int = 0, style: str = "silver") -> str:
    """Draw a rounded box around content."""
    w = width or min(term_width() - 4, 76)
    colors = {
        "silver": (C.S4, C.S6),
        "bright": (C.S1, C.S3),
        "dim":    (C.S7, C.S8),
        "accent": (C.CYAN, C.S5),
    }
    border_c, text_c = colors.get(style, (C.S4, C.S6))

    top_title = f" {title} " if title else ""
    top_pad = w - 2 - len(top_title)
    top = f"{border_c}╭{'─' * (top_pad // 2)}{C.S1}{C.BOLD}{top_title}{C.RST}{border_c}{'─' * (top_pad - top_pad // 2)}╮{C.RST}"
    bot = f"{border_c}╰{'─' * (w - 2)}╯{C.RST}"

    lines = content.split("\n")
    body = []
    for line in lines:
        # Strip ANSI for length calc
        raw = line
        stripped = ""
        i = 0
        while i < len(raw):
            if raw[i] == "\033":
                while i < len(raw) and raw[i] not in "mHJK":
                    i += 1
                i += 1
            else:
                stripped += raw[i]
                i += 1
        pad = w - 2 - len(stripped)
        if pad < 0:
            pad = 0
        body.append(f"{border_c}│{C.RST} {text_c}{line}{C.RST}{' ' * max(0, pad - 1)}{border_c}│{C.RST}")

    return "\n".join([top] + body + [bot])


def progress_bar(pct: float, width: int = 30, filled_char: str = "█", empty_char: str = "░") -> str:
    """Silver-themed progress bar."""
    filled = int(pct * width)
    empty = width - filled

    # Gradient the filled portion
    bar = ""
    for i in range(filled):
        intensity = i / max(width, 1)
        if intensity < 0.3:
            bar += C.S6
        elif intensity < 0.6:
            bar += C.S4
        elif intensity < 0.8:
            bar += C.S2
        else:
            bar += C.S1
        bar += filled_char

    bar += C.S8 + empty_char * empty + C.RST
    return bar


def spinner_frames() -> list[str]:
    return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ─── Animation Engine ──────────────────────────────────────────────────────────

class Animation:
    """Manages terminal animations."""

    @staticmethod
    def boot_sequence():
        """Startup boot animation."""
        clear()
        hide_cursor()

        # Phase 1: Logo shimmer
        for frame in range(20):
            move_to(3, 1)
            logo = render_gradient_logo(frame)
            w = term_width()
            for line in logo.split("\n"):
                # rough center
                stripped_len = len(line.replace("\033", "").split("m")[-1]) if "\033" in line else len(line)
                pad = max(0, (w - 50) // 2)
                print(" " * pad + line)
            time.sleep(0.06)

        # Phase 2: Tagline typewriter
        tagline = "Distributed ML Training Across Apple Silicon"
        move_to(9, 1)
        pad = max(0, (term_width() - len(tagline)) // 2)
        for i, ch in enumerate(tagline):
            sys.stdout.write(f"{C.S3}{ch}")
            sys.stdout.flush()
            if i == 0:
                move_to(9, pad + 1)
            time.sleep(0.02)
        print(C.RST)

        # Phase 3: System check
        move_to(11, 1)
        checks = [
            ("Initializing AirTrain core", True),
            ("Detecting Apple Silicon", True),
            ("Scanning network interfaces", True),
            ("Loading model registry", True),
            ("Connecting to Ollama", None),  # None = will check
        ]

        pad = max(0, (term_width() - 50) // 2)
        frames = spinner_frames()
        for idx, (label, status) in enumerate(checks):
            row = 12 + idx
            # Spinner animation
            for f in range(8):
                move_to(row, pad)
                sys.stdout.write(f"  {C.S3}{frames[f % len(frames)]} {C.S5}{label}...{C.RST}   ")
                sys.stdout.flush()
                time.sleep(0.05)

            # Result
            if status is None:
                # Check Ollama
                status = check_ollama()

            move_to(row, pad)
            icon = f"{C.GREEN}✓" if status else f"{C.YELLOW}○"
            suffix = "" if status else f" {C.S7}(optional)"
            sys.stdout.write(f"  {icon} {C.S3}{label}{suffix}{C.RST}          \n")
            sys.stdout.flush()

        time.sleep(0.5)

        # Phase 4: Ready
        move_to(18, 1)
        ready_msg = f"{C.S1}{C.BOLD}  ■ Ready{C.RST}"
        pad = max(0, (term_width() - 10) // 2)
        move_to(18, pad)
        print(ready_msg)
        time.sleep(0.8)

    @staticmethod
    def training_pulse(step: int, total: int, loss: float, peers: int, toks: float):
        """Animated training step display."""
        pct = step / max(total, 1)
        bar = progress_bar(pct, width=35)

        # Pulsing step indicator
        pulse_chars = ["◇", "◈", "◆", "◈"]
        pulse = pulse_chars[step % len(pulse_chars)]

        line = (
            f"  {C.S3}{pulse} {C.S1}Step {step:>6}{C.S7}/{total}  "
            f"{bar}  "
            f"{C.S3}loss {C.S1}{loss:.4f}  "
            f"{C.S5}│ {C.S3}{peers} {'peers' if peers != 1 else 'peer '}  "
            f"{C.S5}│ {C.S3}{toks:,.0f} tok/s{C.RST}"
        )
        return line

    @staticmethod
    def sync_animation(round_num: int, num_peers: int):
        """Animated gradient sync visualization."""
        w = min(term_width() - 4, 72)
        pad = max(0, (term_width() - w) // 2)

        print()
        print(f"{' ' * pad}{C.S5}{'─' * w}{C.RST}")
        print(f"{' ' * pad}{C.S3}  ⟐  Sync Round {round_num}  ─  {num_peers} peers synchronizing{C.RST}")

        # Gradient flow animation
        flow_chars = "░▒▓█▓▒░"
        for frame in range(12):
            move_to_col = pad + 4
            line = " " * pad + "    "
            for p in range(num_peers):
                offset = (frame + p * 3) % len(flow_chars)
                segment = ""
                for i in range(8):
                    ci = (offset + i) % len(flow_chars)
                    segment += flow_chars[ci]
                arrow = "──►" if p < num_peers - 1 else "──◆"
                colors = [C.S7, C.S6, C.S5, C.S4, C.S3, C.S2, C.S1, C.S2]
                for i, ch in enumerate(segment):
                    line += colors[(frame + i) % len(colors)] + ch
                line += f" {C.S5}{arrow} "
            line += C.RST
            sys.stdout.write(f"\r{line}   ")
            sys.stdout.flush()
            time.sleep(0.08)

        # Completion
        sys.stdout.write(f"\r{' ' * pad}    {C.GREEN}✓ {C.S3}Gradients merged  "
                         f"{C.S5}│  weights broadcast to all peers{C.RST}          \n")
        print(f"{' ' * pad}{C.S5}{'─' * w}{C.RST}")
        print()

    @staticmethod
    def dream_animation(sample_num: int, quality: float, text_preview: str):
        """Dream training visualization."""
        # Quality color
        if quality >= 0.7:
            qc = C.GREEN
            icon = "✦"
        elif quality >= 0.5:
            qc = C.YELLOW
            icon = "◇"
        else:
            qc = C.RED
            icon = "✗"

        qbar = progress_bar(quality, width=12)

        # Truncate preview
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        preview = preview.replace("\n", " ")

        line = (
            f"  {C.S5}dream {C.S3}#{sample_num:<4} "
            f"{qc}{icon}{C.RST} {qbar} {C.S3}{quality:.2f}  "
            f"{C.S7}{C.ITALIC}\"{preview}\"{C.RST}"
        )
        print(line)

    @staticmethod
    def marketplace_display(rankings: list[dict]):
        """Display gradient marketplace rankings."""
        lines = []
        lines.append(f"  {C.S3}{'Rank':<6}{'Peer':<24}{'Weight':>8}{'Mag':>7}{'Align':>7}{'Hist':>7}{'Imp':>7}{C.RST}")
        lines.append(f"  {C.S7}{'─' * 66}{C.RST}")

        medals = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(rankings):
            medal = medals[i] if i < 3 else f"  {C.S6}#{i+1}"
            weight_bar = progress_bar(r["weight"], width=6)

            line = (
                f"  {medal} {C.S3}{r['name']:<22}"
                f"{C.S1}{r['weight']:.3f} {weight_bar} "
                f"{C.S5}{r['mag']:.2f}  {r['align']:.2f}  {r['hist']:.2f}  {r['imp']:.2f}{C.RST}"
            )
            lines.append(line)

        return "\n".join(lines)


# ─── Ollama Integration ───────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def ollama_generate(prompt: str, model: str = "llama3.2", temperature: float = 0.9) -> str | None:
    """Generate text from Ollama."""
    try:
        import urllib.request
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 128}
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except Exception as e:
        return None

def ollama_list_models() -> list[str]:
    """List available Ollama models."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Simulated Peer Data ──────────────────────────────────────────────────────

PEER_NAMES = [
    "MacBook-Pro-Alex", "Mac-Mini-Server", "MacBook-Air-Sam",
    "iMac-Design-Lab", "MacBook-Pro-Jordan", "Mac-Studio-ML",
    "MacBook-Air-Riley", "Mac-Pro-Research"
]

CHIP_TYPES = [
    ("M4 Max", 128, 18.4), ("M4 Pro", 48, 5.3), ("M4", 32, 2.9),
    ("M3 Pro", 36, 4.1), ("M3 Max", 96, 14.2), ("M2 Ultra", 192, 27.6),
    ("M3", 24, 2.5), ("M2 Pro", 32, 3.7),
]

def generate_peers(n: int) -> list[dict]:
    peers = []
    for i in range(n):
        name = PEER_NAMES[i % len(PEER_NAMES)]
        chip, mem, tflops = CHIP_TYPES[i % len(CHIP_TYPES)]
        peers.append({
            "name": name,
            "chip": chip,
            "memory_gb": mem,
            "tflops": tflops,
            "status": "training",
            "ip": f"192.168.1.{10 + i}",
        })
    return peers


# ─── Interactive Shell ─────────────────────────────────────────────────────────

HELP_TEXT = f"""
{C.S1}{C.BOLD}Commands:{C.RST}

  {C.S2}train{C.S5}     Start a distributed training demo
  {C.S2}dream{C.S5}     Run dream training session (uses Ollama if available)
  {C.S2}status{C.S5}    Show cluster status
  {C.S2}peers{C.S5}     List connected peers
  {C.S2}market{C.S5}    Show gradient marketplace rankings
  {C.S2}sync{C.S5}      Show sync animation demo
  {C.S2}relay{C.S5}     Simulate relay checkpoint handoff
  {C.S2}autopsy{C.S5}   Generate model autopsy report
  {C.S2}models{C.S5}    List available Ollama models
  {C.S2}config{C.S5}    Show current configuration
  {C.S2}clear{C.S5}     Clear terminal
  {C.S2}help{C.S5}      Show this help message
  {C.S2}exit{C.S5}      Exit AirTrain

{C.S7}  Tip: Run with --demo for a full automated showcase{C.RST}
"""


def print_header():
    """Print the persistent header."""
    w = term_width()
    ts = datetime.now().strftime("%H:%M:%S")
    status_line = (
        f"  {C.S7}AirTrain v0.1.0{C.RST}"
        f"  {C.S8}│{C.RST}"
        f"  {C.S5}{platform.node()}{C.RST}"
        f"  {C.S8}│{C.RST}"
        f"  {C.S5}{ts}{C.RST}"
    )
    print(f"{C.S8}{'─' * w}{C.RST}")
    print(status_line)
    print(f"{C.S8}{'─' * w}{C.RST}")


def prompt() -> str:
    """Display the command prompt."""
    return f"\n{C.S5}  ❯{C.S1} "


def cmd_train(peers_count: int = 3, steps: int = 50, use_ollama: bool = False):
    """Simulate a distributed training session."""
    peers = generate_peers(peers_count)
    total_tflops = sum(p["tflops"] for p in peers)
    inner_steps = 10

    print()
    info = (
        f"  {C.S3}Model:{C.S1}       GPT-2 Small (124M params)\n"
        f"  {C.S3}Workers:{C.S1}     {len(peers)}\n"
        f"  {C.S3}Combined:{C.S1}    {total_tflops:.1f} TFLOPS\n"
        f"  {C.S3}Inner steps:{C.S1} {inner_steps}\n"
        f"  {C.S3}Total steps:{C.S1} {steps}\n"
        f"  {C.S3}Algorithm:{C.S1}   DiLoCo (500x comm reduction)\n"
        f"  {C.S3}Ollama:{C.S1}      {'connected' if use_ollama else 'simulated'}"
    )
    print(box(info, title="Training Session", style="accent"))
    print()

    loss = 4.8 + random.uniform(-0.2, 0.2)
    start = time.time()

    for step in range(1, steps + 1):
        # Simulate loss decrease with noise
        decay = 0.985 + random.uniform(-0.005, 0.005)
        noise = random.gauss(0, 0.01)
        loss = max(0.5, loss * decay + noise)

        toks = total_tflops * 8500 + random.uniform(-2000, 2000)

        line = Animation.training_pulse(step, steps, loss, len(peers), toks)
        sys.stdout.write(f"\r{line}  ")
        sys.stdout.flush()

        # Sync every inner_steps
        if step % inner_steps == 0 and step < steps:
            print()
            Animation.sync_animation(step // inner_steps, len(peers))

        time.sleep(0.08)

    elapsed = time.time() - start
    print()
    print()

    summary = (
        f"  {C.GREEN}✓ Training complete{C.RST}\n"
        f"\n"
        f"  {C.S3}Final loss:{C.S1}    {loss:.4f}\n"
        f"  {C.S3}Steps:{C.S1}         {steps}\n"
        f"  {C.S3}Time:{C.S1}          {elapsed:.1f}s\n"
        f"  {C.S3}Avg tok/s:{C.S1}     {total_tflops * 8500:,.0f}\n"
        f"  {C.S3}Sync rounds:{C.S1}   {steps // inner_steps}\n"
        f"  {C.S3}Checkpoint:{C.S1}    ./checkpoints/step-{steps}"
    )
    print(box(summary, title="Results", style="bright"))


def cmd_dream(use_ollama: bool = False, model: str = "llama3.2", count: int = 15):
    """Run dream training with or without Ollama."""
    print()
    print(box(
        f"  {C.S3}Mode:{C.S1}       {'Ollama (' + model + ')' if use_ollama else 'Simulated'}\n"
        f"  {C.S3}Samples:{C.S1}    {count}\n"
        f"  {C.S3}Temp:{C.S1}       0.9\n"
        f"  {C.S3}Threshold:{C.S1}  0.55",
        title="Dream Session",
        style="accent"
    ))
    print()

    prompts = [
        "Explain distributed computing in simple terms:",
        "Describe how neural networks learn patterns:",
        "Write about the future of collaborative AI:",
        "Explain gradient descent to a beginner:",
        "Describe the architecture of a transformer model:",
        "Write about the benefits of edge computing:",
        "Explain what makes Apple Silicon efficient:",
        "Describe how federated learning preserves privacy:",
        "Write about the concept of transfer learning:",
        "Explain attention mechanisms in neural networks:",
        "Describe the role of loss functions in training:",
        "Write about distributed training across devices:",
        "Explain backpropagation intuitively:",
        "Describe the unified memory architecture:",
        "Write about the potential of volunteer computing:",
    ]

    kept = 0
    rejected = 0
    qualities = []

    for i in range(count):
        prompt_text = prompts[i % len(prompts)]

        # Show thinking spinner
        frames = spinner_frames()
        for f in range(6 if not use_ollama else 1):
            sys.stdout.write(f"\r  {C.S5}{frames[f % len(frames)]} {C.S7}dreaming...{C.RST}   ")
            sys.stdout.flush()
            time.sleep(0.08)

        if use_ollama:
            text = ollama_generate(prompt_text, model=model, temperature=0.9)
            if not text:
                text = f"[generation failed for: {prompt_text[:30]}...]"
        else:
            # Simulated dream text
            snippets = [
                "The key insight behind distributed training is that gradient computation is embarrassingly parallel",
                "Neural networks learn by adjusting weights through repeated exposure to labeled examples",
                "Transformer models use self-attention to weigh the importance of different input tokens",
                "Apple Silicon's unified memory eliminates the need to copy data between CPU and GPU",
                "Federated learning allows models to improve without centralizing sensitive data",
                "The DiLoCo algorithm reduces communication overhead by synchronizing only every 500 steps",
                "Loss functions measure the discrepancy between predicted and actual outputs",
                "Gradient descent navigates the loss landscape by following the steepest direction of improvement",
                "Transfer learning leverages patterns learned on one task to accelerate learning on another",
                "Backpropagation computes gradients layer by layer using the chain rule of calculus",
                "Attention mechanisms allow models to focus on relevant parts of the input sequence",
                "Edge computing brings computation closer to where data is generated reducing latency",
                "Volunteer computing harnesses idle processing power from personal devices worldwide",
                "The unified memory architecture allows CPU and GPU to share the same memory pool",
                "Collaborative AI training distributes the computational burden across multiple devices",
            ]
            text = snippets[i % len(snippets)]
            time.sleep(0.15)

        # Score
        quality = random.uniform(0.3, 0.95)
        # Bias toward good quality for ollama
        if use_ollama and text and "[generation failed" not in text:
            quality = min(1.0, quality + 0.15)

        qualities.append(quality)
        sys.stdout.write("\r" + " " * 60 + "\r")
        Animation.dream_animation(i + 1, quality, text)

        if quality >= 0.55:
            kept += 1
        else:
            rejected += 1

    avg_q = sum(qualities) / len(qualities) if qualities else 0

    print()
    result = (
        f"  {C.S3}Generated:{C.S1}   {count}\n"
        f"  {C.GREEN}Kept:{C.S1}        {kept} ({kept/count*100:.0f}%)\n"
        f"  {C.RED}Rejected:{C.S1}    {rejected} ({rejected/count*100:.0f}%)\n"
        f"  {C.S3}Avg quality:{C.S1} {avg_q:.3f}\n"
        f"  {C.S3}Cache:{C.S1}       {kept} samples ({kept * 2:.1f} KB)"
    )
    print(box(result, title="Dream Results", style="bright"))


def cmd_status(peers_count: int = 3):
    """Show cluster status."""
    peers = generate_peers(peers_count)
    total_tflops = sum(p["tflops"] for p in peers)
    total_mem = sum(p["memory_gb"] for p in peers)

    print()
    status = (
        f"  {C.S3}Status:{C.GREEN}      ● Training{C.RST}\n"
        f"  {C.S3}Model:{C.S1}       GPT-2 Small (124M)\n"
        f"  {C.S3}Step:{C.S1}        14,832 / 100,000\n"
        f"  {C.S3}Loss:{C.S1}        2.847\n"
        f"  {C.S3}Peers:{C.S1}       {len(peers)}\n"
        f"  {C.S3}Compute:{C.S1}     {total_tflops:.1f} TFLOPS combined\n"
        f"  {C.S3}Memory:{C.S1}      {total_mem} GB unified\n"
        f"  {C.S3}Throughput:{C.S1}  {total_tflops * 8500:,.0f} tok/s\n"
        f"  {C.S3}Uptime:{C.S1}      3h 42m\n"
        f"  {C.S3}Sync round:{C.S1}  29 / ∞\n"
        f"  {C.S3}Next sync:{C.S1}   in ~168 steps\n"
        f"  {C.S3}Checkpoint:{C.S1}  ./checkpoints/step-14000"
    )
    print(box(status, title="Cluster Status", style="accent"))


def cmd_peers(peers_count: int = 3):
    """List connected peers."""
    peers = generate_peers(peers_count)

    print()
    lines = []
    lines.append(f"  {C.S3}{'Peer':<24}{'Chip':<12}{'Mem':>6}{'TFLOPS':>9}{'Status':>10}{C.RST}")
    lines.append(f"  {C.S7}{'─' * 62}{C.RST}")

    for p in peers:
        status_color = C.GREEN if p["status"] == "training" else C.YELLOW
        lines.append(
            f"  {C.S1}{p['name']:<24}"
            f"{C.S3}{p['chip']:<12}"
            f"{C.S5}{p['memory_gb']:>4} GB"
            f"{C.S3}{p['tflops']:>8.1f}"
            f"  {status_color}● {p['status']}{C.RST}"
        )

    total_tflops = sum(p["tflops"] for p in peers)
    lines.append(f"  {C.S7}{'─' * 62}{C.RST}")
    lines.append(f"  {C.S3}{'Total':<36}{C.S1}{sum(p['memory_gb'] for p in peers):>4} GB{total_tflops:>8.1f}{C.RST}")

    print(box("\n".join(lines), title=f"Connected Peers ({len(peers)})", style="silver"))


def cmd_market(peers_count: int = 4):
    """Show marketplace rankings."""
    peers = generate_peers(peers_count)
    rankings = []
    weights_raw = []
    for p in peers:
        mag = random.uniform(0.5, 0.98)
        align = random.uniform(0.5, 0.95)
        hist = random.uniform(0.4, 0.9)
        imp = random.uniform(0.4, 0.85)
        total = mag * 0.25 + align * 0.35 + hist * 0.25 + imp * 0.15
        rankings.append({
            "name": p["name"], "mag": mag, "align": align,
            "hist": hist, "imp": imp, "total": total, "weight": 0
        })
        weights_raw.append(total)

    # Normalize weights
    total_w = sum(weights_raw)
    for i, r in enumerate(rankings):
        r["weight"] = max(0.1, weights_raw[i] / total_w)

    # Re-normalize after floor
    total_w2 = sum(r["weight"] for r in rankings)
    for r in rankings:
        r["weight"] /= total_w2

    rankings.sort(key=lambda x: -x["weight"])

    print()
    display = Animation.marketplace_display(rankings)
    print(box(display, title="Gradient Marketplace — Round 12", style="accent"))


def cmd_relay():
    """Simulate relay checkpoint handoff."""
    print()
    frames = spinner_frames()

    steps_data = [
        ("Packaging model weights", "127.4 MB (safetensors)"),
        ("Saving optimizer state", "254.8 MB (npz)"),
        ("Writing metadata", "2.1 KB (json)"),
        ("Compressing checkpoint", "89.2 MB total"),
        ("Generating relay manifest", "ready to share"),
    ]

    pad = max(0, (term_width() - 60) // 2)
    print(f"{' ' * pad}{C.S3}  Exporting relay checkpoint...{C.RST}")
    print()

    for label, detail in steps_data:
        for f in range(10):
            move_col = pad
            sys.stdout.write(f"\r{' ' * pad}  {C.S5}{frames[f % len(frames)]} {C.S3}{label}...{C.RST}   ")
            sys.stdout.flush()
            time.sleep(0.06)

        sys.stdout.write(f"\r{' ' * pad}  {C.GREEN}✓ {C.S3}{label}  {C.S7}{detail}{C.RST}          \n")

    print()
    result = (
        f"  {C.GREEN}✓ Relay checkpoint exported{C.RST}\n"
        f"\n"
        f"  {C.S3}Location:{C.S1}      ./relay-gpt2-step14832/\n"
        f"  {C.S3}Model:{C.S1}         GPT-2 Small (124M)\n"
        f"  {C.S3}Step:{C.S1}          14,832\n"
        f"  {C.S3}Loss:{C.S1}          2.847\n"
        f"  {C.S3}Contributors:{C.S1}  3\n"
        f"  {C.S3}Compute hrs:{C.S1}   4.2\n"
        f"  {C.S3}Size:{C.S1}          89.2 MB\n"
        f"\n"
        f"  {C.S5}Share this folder. The next trainer runs:{C.RST}\n"
        f"  {C.S1}airtrain start --resume ./relay-gpt2-step14832{C.RST}"
    )
    print(box(result, title="Relay Export", style="bright"))


def cmd_autopsy():
    """Generate a simulated autopsy report."""
    print()
    frames = spinner_frames()
    pad = max(0, (term_width() - 50) // 2)

    steps = [
        "Analyzing training events",
        "Computing contributor rankings",
        "Identifying breakthrough rounds",
        "Measuring dream impact",
        "Generating loss curve data",
        "Rendering HTML report",
    ]

    for label in steps:
        for f in range(8):
            sys.stdout.write(f"\r{' ' * pad}  {C.S5}{frames[f % len(frames)]} {C.S3}{label}...{C.RST}   ")
            sys.stdout.flush()
            time.sleep(0.05)
        sys.stdout.write(f"\r{' ' * pad}  {C.GREEN}✓ {C.S3}{label}{C.RST}                    \n")

    print()

    # Summary
    report = (
        f"  {C.S1}{C.BOLD}Model Autopsy: GPT-2 Small{C.RST}\n"
        f"\n"
        f"  {C.S3}Total steps:{C.S1}        50,000\n"
        f"  {C.S3}Sync rounds:{C.S1}        100\n"
        f"  {C.S3}Initial loss:{C.S1}       4.82\n"
        f"  {C.S3}Final loss:{C.S1}         1.94\n"
        f"  {C.S3}Loss reduction:{C.S1}     {C.GREEN}59.8%{C.RST}\n"
        f"  {C.S3}Compute hours:{C.S1}      18.4\n"
        f"  {C.S3}Contributors:{C.S1}       5\n"
        f"  {C.S3}Dream samples:{C.S1}      2,847 generated, 1,923 kept\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Top Contributors:{C.RST}\n"
        f"  {C.S3}  1. MacBook-Pro-Alex    {C.S1}6.2 hrs  {C.S5}42 syncs{C.RST}\n"
        f"  {C.S3}  2. Mac-Mini-Server     {C.S1}5.8 hrs  {C.S5}38 syncs{C.RST}\n"
        f"  {C.S3}  3. MacBook-Air-Sam     {C.S1}3.1 hrs  {C.S5}20 syncs{C.RST}\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Breakthrough Rounds:{C.RST}\n"
        f"  {C.S3}  Round 14:{C.S1} loss 3.91→3.42 {C.GREEN}(-12.5%){C.S5}  3 peers{C.RST}\n"
        f"  {C.S3}  Round 38:{C.S1} loss 2.67→2.31 {C.GREEN}(-13.5%){C.S5}  4 peers{C.RST}\n"
        f"  {C.S3}  Round 71:{C.S1} loss 2.12→1.94 {C.GREEN}( -8.5%){C.S5}  5 peers{C.RST}\n"
        f"\n"
        f"  {C.S3}Report:{C.S1} ./autopsy/report.html{C.RST}"
    )
    print(box(report, title="Autopsy Report", style="bright"))


def cmd_config():
    """Show current configuration."""
    print()
    config = (
        f"  {C.S2}{C.BOLD}Training{C.RST}\n"
        f"  {C.S3}model_name:{C.S1}         gpt2-small\n"
        f"  {C.S3}batch_size:{C.S1}         8\n"
        f"  {C.S3}max_steps:{C.S1}          100,000\n"
        f"  {C.S3}seq_length:{C.S1}         512\n"
        f"  {C.S3}checkpoint_every:{C.S1}   1,000\n"
        f"\n"
        f"  {C.S2}{C.BOLD}DiLoCo{C.RST}\n"
        f"  {C.S3}inner_steps:{C.S1}        500\n"
        f"  {C.S3}inner_lr:{C.S1}           3e-4 (AdamW)\n"
        f"  {C.S3}outer_lr:{C.S1}           0.7 (SGD+Nesterov)\n"
        f"  {C.S3}outer_momentum:{C.S1}     0.9\n"
        f"  {C.S3}gradient_compress:{C.S1}  FP16 + gzip\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Network{C.RST}\n"
        f"  {C.S3}port:{C.S1}              7471\n"
        f"  {C.S3}discovery:{C.S1}         mDNS (Bonjour)\n"
        f"  {C.S3}heartbeat:{C.S1}         5.0s\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Marketplace{C.RST}\n"
        f"  {C.S3}magnitude:{C.S1}         0.25\n"
        f"  {C.S3}alignment:{C.S1}         0.35\n"
        f"  {C.S3}history:{C.S1}           0.25\n"
        f"  {C.S3}improvement:{C.S1}       0.15\n"
        f"  {C.S3}min_weight:{C.S1}        0.10\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Sleep Swarm{C.RST}\n"
        f"  {C.S3}window:{C.S1}            23:00 - 07:00\n"
        f"  {C.S3}min_battery:{C.S1}       20%\n"
        f"\n"
        f"  {C.S2}{C.BOLD}Dreams{C.RST}\n"
        f"  {C.S3}samples:{C.S1}           1,000 per session\n"
        f"  {C.S3}temperature:{C.S1}       0.9\n"
        f"  {C.S3}quality_threshold:{C.S1} 0.55\n"
        f"  {C.S3}mix_ratio:{C.S1}         15%"
    )
    print(box(config, title="Configuration", style="silver"))


def cmd_models():
    """List Ollama models."""
    models = ollama_list_models()
    print()
    if models:
        lines = []
        for m in models:
            lines.append(f"  {C.S3}●{C.S1} {m}{C.RST}")
        print(box("\n".join(lines), title=f"Ollama Models ({len(models)})", style="accent"))
    else:
        print(box(
            f"  {C.YELLOW}No Ollama models found.{C.RST}\n"
            f"  {C.S5}Install Ollama and pull a model:{C.RST}\n"
            f"  {C.S3}  brew install ollama{C.RST}\n"
            f"  {C.S3}  ollama pull llama3.2{C.RST}",
            title="Ollama Models",
            style="dim"
        ))


# ─── Demo Mode ─────────────────────────────────────────────────────────────────

def run_demo():
    """Full automated demo sequence."""
    has_ollama = check_ollama()

    Animation.boot_sequence()
    time.sleep(1)
    clear()

    # Show logo
    print()
    w = term_width()
    for line in LOGO_SMALL.split("\n"):
        pad = max(0, (w - 50) // 2)
        print(" " * pad + line)
    print()
    print_header()

    # Status
    print(f"\n  {C.S2}{C.BOLD}━━━ Cluster Status ━━━{C.RST}")
    cmd_status(4)
    time.sleep(1.5)

    # Peers
    print(f"\n  {C.S2}{C.BOLD}━━━ Connected Peers ━━━{C.RST}")
    cmd_peers(4)
    time.sleep(1.5)

    # Training
    print(f"\n  {C.S2}{C.BOLD}━━━ Training Demo ━━━{C.RST}")
    cmd_train(peers_count=4, steps=30, use_ollama=False)
    time.sleep(1.5)

    # Marketplace
    print(f"\n  {C.S2}{C.BOLD}━━━ Gradient Marketplace ━━━{C.RST}")
    cmd_market(4)
    time.sleep(1.5)

    # Dreams
    print(f"\n  {C.S2}{C.BOLD}━━━ Dream Training ━━━{C.RST}")
    cmd_dream(use_ollama=has_ollama, count=8)
    time.sleep(1.5)

    # Relay
    print(f"\n  {C.S2}{C.BOLD}━━━ Relay Checkpoint ━━━{C.RST}")
    cmd_relay()
    time.sleep(1.5)

    # Autopsy
    print(f"\n  {C.S2}{C.BOLD}━━━ Model Autopsy ━━━{C.RST}")
    cmd_autopsy()
    time.sleep(1)

    # Finale
    print()
    print()
    w = term_width()
    for line in LOGO_SMALL.split("\n"):
        pad = max(0, (w - 50) // 2)
        print(" " * pad + line)

    pad = max(0, (w - 50) // 2)
    print()
    print(f"{' ' * pad}{C.S3}  github.com/alexandercodes4/AirTrain{C.RST}")
    print(f"{' ' * pad}{C.S5}  Distributed ML training. Free. Open source.{C.RST}")
    print()

    show_cursor()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AirTrain Interactive CLI")
    parser.add_argument("--demo", action="store_true", help="Run full demo sequence")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for dream training")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def cleanup(sig=None, frame=None):
        show_cursor()
        print(f"\n{C.S5}  Goodbye.{C.RST}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    if args.demo:
        run_demo()
        return

    # Boot sequence
    Animation.boot_sequence()
    time.sleep(0.5)
    clear()

    # Logo
    print()
    w = term_width()
    for line in LOGO_SMALL.split("\n"):
        pad = max(0, (w - 50) // 2)
        print(" " * pad + line)
    print()
    print_header()

    has_ollama = check_ollama()
    use_ollama = has_ollama and args.ollama

    print(f"\n  {C.S5}Type {C.S1}help{C.S5} for commands.{C.RST}")

    # REPL
    while True:
        try:
            show_cursor()
            raw = input(prompt())
            print(C.RST, end="")
            cmd = raw.strip().lower()

            if not cmd:
                continue
            elif cmd in ("exit", "quit", "q"):
                cleanup()
            elif cmd == "help":
                print(HELP_TEXT)
            elif cmd == "clear":
                clear()
                print()
                for line in LOGO_SMALL.split("\n"):
                    pad = max(0, (w - 50) // 2)
                    print(" " * pad + line)
                print()
                print_header()
            elif cmd == "train":
                hide_cursor()
                cmd_train(peers_count=3, steps=50, use_ollama=use_ollama)
            elif cmd.startswith("dream"):
                hide_cursor()
                count = 15
                parts = cmd.split()
                if len(parts) > 1 and parts[1].isdigit():
                    count = int(parts[1])
                cmd_dream(use_ollama=use_ollama or has_ollama, model=args.model, count=count)
            elif cmd == "status":
                cmd_status()
            elif cmd == "peers":
                cmd_peers()
            elif cmd == "market":
                cmd_market()
            elif cmd == "sync":
                hide_cursor()
                Animation.sync_animation(random.randint(1, 50), random.randint(2, 5))
            elif cmd == "relay":
                hide_cursor()
                cmd_relay()
            elif cmd == "autopsy":
                hide_cursor()
                cmd_autopsy()
            elif cmd == "models":
                cmd_models()
            elif cmd == "config":
                cmd_config()
            else:
                print(f"  {C.S7}Unknown command: {cmd}. Type {C.S3}help{C.S7} for commands.{C.RST}")

        except EOFError:
            cleanup()
        except KeyboardInterrupt:
            cleanup()

    show_cursor()


if __name__ == "__main__":
    main()
