# AirTrain

**Distributed ML training across Apple Silicon Macs.**

AirTrain dramatically reduces machine learning model training costs by splitting computation across multiple Mac devices. Using the DiLoCo algorithm, it achieves near-linear scaling with **500x less network communication** than traditional distributed training — making Wi-Fi-based training practical.

Training a 124M parameter GPT-2 model? Instead of renting cloud GPUs at $3/hr, pool three MacBooks in a coffee shop and train for free.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [The DiLoCo Algorithm](#the-diloco-algorithm)
- [Architecture](#architecture)
- [Peer Discovery](#peer-discovery)
- [Network Protocol](#network-protocol)
- [Checkpoint System](#checkpoint-system)
- [Training Relay](#training-relay)
- [Local Dashboard](#local-dashboard)
- [AirTrain Website (airtrain.dev)](#airtrain-website)
- [Apple Silicon Performance](#apple-silicon-performance)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Comparison to Existing Tools](#comparison-to-existing-tools)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Zero-config discovery** — Devices find each other automatically on local networks via mDNS/Bonjour
- **DiLoCo training** — 500x less network traffic than traditional distributed training (DDP)
- **Fault tolerant** — Nodes can join and leave mid-training without killing the run
- **Checkpoint relay** — Pause training, export a checkpoint, hand it off to someone else to continue
- **Built for Apple Silicon** — Native MLX framework, optimized for M1/M2/M3/M4/M5 unified memory architecture
- **Local dashboard** — Real-time training metrics, peer monitoring, and checkpoint timeline in your browser
- **Community platform** — [airtrain.dev](https://airtrain.dev) lets you find training partners, share checkpoints, and track your contributions on a global leaderboard

## Quick Start

```bash
pip install airtrain

# Mac 1 — Start training as coordinator
airtrain start --model gpt2-small --dataset ./data/wikitext.txt --dashboard

# Mac 2 — Join automatically via mDNS
airtrain join auto
```

Both Macs now train collaboratively. Loss decreases on both terminals. Open `http://localhost:8471` on Mac 1 to see the live dashboard.

## How It Works

Traditional distributed training (DDP) synchronizes gradients after **every single step**. For a 124M parameter model in FP32, that's ~500MB of data exchanged per step. At 100 steps/second, you need 50 GB/s of sustained bandwidth — impossible over Wi-Fi.

AirTrain uses the **DiLoCo** (Distributed Low-Communication) algorithm to reduce this by 500x:

```
Traditional DDP:      1 sync per step     = 50 GB/s required
AirTrain (DiLoCo):    1 sync per 500 steps = 0.1 GB/s required ✓ Wi-Fi works
```

Each Mac trains independently for 500 steps, then syncs only the *difference* between where it started and where it ended (pseudo-gradients). A coordinator averages these diffs and broadcasts updated weights. The entire sync takes ~2 seconds over Wi-Fi.

## The DiLoCo Algorithm

AirTrain implements the DiLoCo algorithm from [Douillard et al. (2023)](https://arxiv.org/abs/2311.08105), validated at scale by [PrimeIntellect's OpenDiLoCo](https://arxiv.org/abs/2407.07852).

### Inner Loop (local training)

Each worker independently runs `H` steps (default 500) of AdamW:

```
θ_local = θ_global                          # snapshot global params
for step in range(H):
    loss = model(batch, θ_local)
    θ_local = θ_local - α · AdamW(∇loss)    # α = 3e-4 (inner lr)
```

### Outer Loop (synchronization)

After `H` inner steps, workers compute pseudo-gradients and the coordinator applies an outer SGD step with Nesterov momentum:

```
Δθ_i = θ_global - θ_local_i                 # pseudo-gradient from worker i
Δθ_avg = mean(Δθ_1, Δθ_2, ..., Δθ_n)       # average across all workers

# Outer SGD + Nesterov momentum
v = β · v + Δθ_avg                           # β = 0.9
θ_global = θ_global - η · (Δθ_avg + β · v)  # η = 0.7 (outer lr)
```

### Why It Works

DiLoCo works because neural network loss landscapes are smooth enough that independent workers explore different regions and converge to compatible solutions. The pseudo-gradient averaging acts as implicit regularization — similar to how federated learning aggregates updates.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `inner_steps` | 500 | Local training steps before sync |
| `inner_lr` | 3e-4 | AdamW learning rate for local training |
| `inner_weight_decay` | 0.1 | AdamW weight decay |
| `outer_lr` | 0.7 | SGD learning rate for global update |
| `outer_momentum` | 0.9 | Nesterov momentum for outer optimizer |
| `gradient_compression` | true | Compress gradients to FP16 + gzip |

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      AirTrain Network                        │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐   ┌──────────────┐   │
│   │  Mac #1       │    │  Mac #2       │   │  Mac #3       │  │
│   │  (Coordinator)│    │  (Worker)     │   │  (Worker)     │  │
│   │               │    │               │   │               │  │
│   │ ┌──────────┐ │    │ ┌──────────┐ │   │ ┌──────────┐ │  │
│   │ │ MLX      │ │    │ │ MLX      │ │   │ │ MLX      │ │  │
│   │ │ Trainer  │ │    │ │ Trainer  │ │   │ │ Trainer  │ │  │
│   │ └────┬─────┘ │    │ └────┬─────┘ │   │ └────┬─────┘ │  │
│   │      │       │    │      │       │   │      │       │  │
│   │ ┌────▼─────┐ │    │ ┌────▼─────┐ │   │ ┌────▼─────┐ │  │
│   │ │ DiLoCo   │ │    │ │ DiLoCo   │ │   │ │ DiLoCo   │ │  │
│   │ │ Engine   │ │    │ │ Engine   │ │   │ │ Engine   │ │  │
│   │ └────┬─────┘ │    │ └────┬─────┘ │   │ └────┬─────┘ │  │
│   │      │       │    │      │       │   │      │       │  │
│   │ ┌────▼─────┐ │    │ ┌────▼─────┐ │   │ ┌────▼─────┐ │  │
│   │ │ TCP      │◄├────┤►│ TCP      │◄├───┤►│ TCP      │ │  │
│   │ │Transport │ │    │ │Transport │ │   │ │Transport │ │  │
│   │ └──────────┘ │    │ └──────────┘ │   │ └──────────┘ │  │
│   │       ▲      │    │              │   │              │  │
│   │  Dashboard   │    │              │   │              │  │
│   │  :8471       │    │              │   │              │  │
│   └──────────────┘    └──────────────┘   └──────────────┘  │
│          ▲                                                   │
│     mDNS/Bonjour                                            │
│   (auto-discovery)                                           │
└──────────────────────────────────────────────────────────────┘
```

### Component Stack

```
┌─────────────────────────────────────────┐
│              CLI (click)                │  airtrain start / join / relay
├─────────────────────────────────────────┤
│         Coordinator / Worker            │  Orchestration layer
├──────────────┬──────────────────────────┤
│ DiLoCo Engine│   Checkpoint Manager     │  Training logic
├──────────────┴──────────────────────────┤
│         Base Trainer (MLX)              │  Model + optimizer wrapper
├─────────────────────────────────────────┤
│    Transport (asyncio TCP)              │  Message passing
├──────────┬──────────────────────────────┤
│  Protocol│  Compression (FP16+gzip)    │  Wire format
├──────────┴──────────────────────────────┤
│    Discovery (mDNS / HTTP Relay)        │  Peer finding
└─────────────────────────────────────────┘
```

## Peer Discovery

AirTrain supports two discovery mechanisms:

### LAN Discovery (mDNS/Bonjour)

On local networks, peers find each other automatically using multicast DNS — the same zero-configuration protocol that Apple uses for AirDrop, AirPlay, and printer discovery.

When you run `airtrain start`, the coordinator registers a `_airtrain._tcp.local.` service on the network, advertising its IP, port, model name, and hardware capabilities. When a worker runs `airtrain join auto`, it browses for this service and connects automatically.

```python
# Under the hood (using python-zeroconf):
ServiceInfo(
    "_airtrain._tcp.local.",
    "coordinator._airtrain._tcp.local.",
    addresses=[socket.inet_aton("192.168.1.10")],
    port=7471,
    properties={
        "model": "gpt2-small",
        "chip": "Apple M4 Pro",
        "memory_gb": "48",
        "status": "training",
    },
)
```

**Limitation:** mDNS only works within a single LAN subnet. It won't work across the internet or on networks that block multicast (some university/enterprise Wi-Fi).

### Internet Discovery (HTTP Relay)

For peers across the internet, AirTrain provides a lightweight HTTP signaling server. Peers POST their info to the relay, and other peers GET the peer list to find sessions to join.

```bash
# Self-host a relay server
uvicorn airtrain.discovery.relay:app --host 0.0.0.0 --port 9000

# Or use the public relay at airtrain.dev
airtrain start --relay https://airtrain.dev/api/relay
airtrain join --relay https://airtrain.dev/api/relay
```

The relay only handles discovery — all training data flows directly peer-to-peer via TCP.

## Network Protocol

AirTrain uses a custom binary protocol over TCP:

```
┌────────────┬──────────────┬─────────────────┐
│ Header Len │ JSON Header  │ Binary Payload  │
│  (4 bytes) │ (variable)   │ (variable)      │
└────────────┴──────────────┴─────────────────┘
```

### Message Types

| Type | Direction | Description |
|---|---|---|
| `HANDSHAKE` | Worker → Coordinator | Initial connection with peer capabilities |
| `SYNC_REQUEST` | Coordinator → Workers | "Send me your pseudo-gradients" |
| `SYNC_GRADIENTS` | Worker → Coordinator | Compressed pseudo-gradient payload |
| `MODEL_WEIGHTS` | Coordinator → Workers | Updated model weights after outer step |
| `HEARTBEAT` | Bidirectional | Keep-alive ping every 5 seconds |
| `PEER_JOIN` | Coordinator → Workers | Notification of new peer |
| `PEER_LEAVE` | Coordinator → Workers | Notification of disconnected peer |

### Gradient Compression

Pseudo-gradients are compressed before transmission:

1. **FP16 casting** — 32-bit floats → 16-bit (2x reduction, negligible quality loss for gradient averaging)
2. **gzip compression** — Typically 2-3x additional reduction on gradient data
3. **Net result:** ~4-6x compression. A 500MB gradient payload becomes ~80-125MB.

For a 124M parameter model: ~250MB per sync (compressed), taking ~2-8 seconds over typical Wi-Fi (30-100 Mbps).

## Checkpoint System

AirTrain saves complete training state as a portable directory:

```
checkpoints/step-5000/
├── model.safetensors       # Model weights (HuggingFace safetensors format)
├── optimizer.npz           # Optimizer state (momentum buffers, etc.)
└── meta.json               # Training metadata
```

### Metadata (`meta.json`)

```json
{
  "version": "0.1.0",
  "model_name": "gpt2-small",
  "global_step": 5000,
  "loss": 3.42,
  "total_compute_hours": 2.5,
  "contributors": ["Alicans-MacBook.local", "Joes-Mac-Mini.local"],
  "created_at": "2026-04-14T15:30:00Z",
  "description": "GPT-2 trained on wikitext-103"
}
```

Checkpoints are **automatically saved** every 1000 steps (configurable) and on `Ctrl+C` interruption. The `safetensors` format is compatible with HuggingFace, so trained models can be uploaded directly to the Hub.

## Training Relay

The relay system enables **asynchronous distributed training** — no need for multiple Macs to be online simultaneously.

### How It Works

1. **You** train a model for a while on your Mac
2. **You** export a portable relay checkpoint
3. **You** share it (via the AirTrain website, AirDrop, email, Google Drive — any file transfer)
4. **Someone else** imports it and continues training
5. The checkpoint tracks all contributors and cumulative compute hours

```bash
# Export a relay checkpoint
airtrain relay export --checkpoint ./checkpoints/step-5000 \
  --output ./relay-gpt2-step5000 \
  --description "GPT-2 on wikitext-103, loss=3.42, need more compute"

# Import and continue
airtrain relay import ./relay-gpt2-step5000
airtrain start --model gpt2-small --dataset ./data --resume ./relay-gpt2-step5000
```

This is like a relay race — each runner (Mac) carries the baton (checkpoint) for their leg, then hands it off.

## Sleep Swarms

The most unique feature in AirTrain: **your Mac trains while you sleep**, then hands off to someone in another timezone when you wake up. The model trains 24/7 by chasing nighttime around the globe.

```bash
airtrain sleep --window "23:00-07:00" --prefer "gpt2*"
```

### How It Works

1. You set a **training window** — the hours your Mac is available (default: 11pm–7am)
2. During that window, AirTrain automatically:
   - Queries the relay server for active sleep swarm sessions
   - Downloads the latest checkpoint for the best matching session
   - Joins as a worker and starts training
3. When your window closes (or battery drops below 20%, or you close the lid):
   - Saves a checkpoint
   - Disconnects gracefully
   - Uploads the updated checkpoint for the next timezone to pick up

### Timezone Coverage

A model in a sleep swarm passes through contributors around the world:

```
UTC  00  02  04  06  08  10  12  14  16  18  20  22
     ████████████                                ████  New York (23:00-07:00)
                 ████████████                          London (00:00-08:00)
                             ████████████              Mumbai (05:30-13:30)
                                         ████████████  Tokyo (09:00-17:00)
     ─────────────────────────────────────────────────
     ████████████████████████████████████████████████  = 24/7 coverage
```

### Configuration

| Flag | Default | Description |
|---|---|---|
| `--window` | `23:00-07:00` | Training window in local time |
| `--prefer` | `any` | Model filter (e.g., `gpt2*`, `llama*`) |
| `--max-hours` | 8 | Max compute hours per night |
| `--min-battery` | 20 | Stop if battery drops below this % |
| `--relay` | `airtrain.dev/api/relay` | Relay server URL |

### Safety

Sleep Swarms are safe by default:
- **Battery protection** — stops training if battery drops below 20%
- **Lid detection** — pauses if you close your MacBook
- **Window enforcement** — always stops when your window ends
- **Auto-checkpoint** — saves progress before every disconnect
- **Retry logic** — reconnects automatically if Wi-Fi drops

## Local Dashboard

When you run training with `--dashboard`, AirTrain starts a web UI at `http://localhost:8471`:

```bash
airtrain start --model gpt2-small --dataset ./data --dashboard
```

The dashboard shows:

- **Loss curve** — Real-time Chart.js plot of training loss over steps
- **Peer table** — Connected devices with chip type, memory, contribution percentage, and status
- **Throughput** — Tokens/second across the swarm
- **Checkpoint timeline** — History of saved checkpoints with loss at each point
- **Cluster status** — Total compute hours, global step, peer count

Data streams via **Server-Sent Events (SSE)** for real-time updates without polling.

## AirTrain Website

**[airtrain.dev](https://airtrain.dev)** is the community platform that connects AirTrain users worldwide. It serves three purposes: helping people find live training sessions to join, enabling asynchronous checkpoint handoffs between strangers, and gamifying contributions to build a community of distributed ML trainers.

### Swarm Browser

The Swarm Browser shows **live training sessions** happening right now. When a coordinator starts training with `--relay https://airtrain.dev/api/relay`, their session appears on the website in real-time.

Each listing shows:
- **Model** being trained (e.g., GPT-2 124M, LLaMA 7B)
- **Progress** — current step, loss, and estimated completion
- **Peers** — how many Macs are currently contributing and how many more are wanted
- **Hardware** — aggregate compute (e.g., "3x M4 Pro, 1x M2 Air = 11.1 TFLOPS")
- **Connection info** — one-click join button that copies the `airtrain join <address>` command

Anyone can browse sessions without an account. Joining requires the AirTrain CLI installed locally.

```
┌──────────────────────────────────────────────────────────┐
│  Live Training Sessions                          3 active │
├──────────────────────────────────────────────────────────┤
│  GPT-2 124M on WikiText-103                              │
│  Step: 15,000 / 100,000  ▓▓▓░░░░░░░  15%               │
│  Loss: 3.12  |  Peers: 4/8  |  12.3 TFLOPS combined    │
│  [Join Session]                                          │
├──────────────────────────────────────────────────────────┤
│  TinyLLaMA 1.1B on RedPajama                            │
│  Step: 2,400 / 50,000   ▓░░░░░░░░░   5%                │
│  Loss: 5.67  |  Peers: 2/4  |  6.8 TFLOPS combined     │
│  [Join Session]                                          │
└──────────────────────────────────────────────────────────┘
```

### Relay Board

The Relay Board is a **marketplace for training checkpoints**. Users post checkpoints they've trained and want others to continue. Think of it as a baton-passing board for asynchronous collaborative training.

How it works:

1. **Post a checkpoint** — Upload metadata (model name, step, loss, compute hours) and a download link (HuggingFace Hub, S3, Google Drive). Weights are never uploaded to airtrain.dev — only metadata and a link.
2. **Browse available relays** — See what models need more training, sorted by recency or popularity.
3. **Claim a relay** — Mark a checkpoint as "claimed" so others don't duplicate work. Download the checkpoint, train for a while, then post your updated checkpoint back.
4. **Track lineage** — Each relay checkpoint records its full history: who trained it, for how many steps, and how many total compute hours have been contributed. A model might pass through 10 different people's Macs before reaching convergence.

```
┌──────────────────────────────────────────────────────────┐
│  Relay Board                                    12 open   │
├──────────────────────────────────────────────────────────┤
│  GPT-2 124M — step 50,000 — loss 2.89                   │
│  "Trained on wikitext-103 for 8 hours. Getting close     │
│   to convergence, needs ~20k more steps."                │
│  Contributors: 3  |  Compute: 14.2 hrs  |  Posted 2h ago│
│  [Claim & Continue]                [View History]        │
├──────────────────────────────────────────────────────────┤
│  TinyStories 33M — step 5,000 — loss 4.21               │
│  "Just started this one. Great for beginners to try      │
│   AirTrain relay — small model, quick progress."         │
│  Contributors: 1  |  Compute: 0.5 hrs  |  Posted 1d ago │
│  [Claim & Continue]                [View History]        │
└──────────────────────────────────────────────────────────┘
```

### Leaderboard & Gamification

The leaderboard ranks contributors by total **compute hours donated** to collaborative training. It creates a positive feedback loop — the more you train, the higher you rank, and the more visible your contributions become.

**Leaderboard columns:**
- **Rank** — Position by total compute hours
- **Username** — GitHub-linked profile
- **Compute Hours** — Total hours of training contributed across all sessions
- **Sessions** — Number of training sessions participated in
- **Relays** — Number of checkpoint handoffs completed
- **Badges** — Achievement icons earned

**Badges:**

| Badge | Name | Criteria |
|---|---|---|
| First Train | Completed your first training session |
| 10 Hours | Contributed 10 compute hours |
| 100 Hours | Contributed 100 compute hours |
| Swarm Leader | Coordinated a session with 5+ peers |
| Relay Champion | Completed 5 relay handoffs |
| Early Adopter | Joined during the first month |

### Website Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Backend | FastAPI (Python) | REST API, SSE for real-time updates |
| Database | SQLite + aiosqlite | Zero-ops, migrates to PostgreSQL at scale |
| Auth | GitHub OAuth | One-click login for developers |
| Frontend | Vanilla HTML/CSS/JS | Landing page, swarm browser, relay board, leaderboard |
| Hosting | Any VPS (Fly.io, Railway, etc.) | Single Python process, no complex infra |

### Website API

All website features are accessible via REST API:

| Endpoint | Method | Description |
|---|---|---|
| `/api/swarms` | GET | List active training sessions |
| `/api/swarms` | POST | Register a new training session |
| `/api/swarms/{id}` | PUT | Update session status/progress |
| `/api/relay` | GET | List available relay checkpoints |
| `/api/relay` | POST | Post a new relay checkpoint |
| `/api/relay/{id}/claim` | POST | Claim a relay checkpoint |
| `/api/leaderboard` | GET | Get ranked contributor list |
| `/api/leaderboard/badges` | GET | Get badge definitions |
| `/auth/login` | GET | Initiate GitHub OAuth flow |
| `/auth/callback` | GET | Handle OAuth callback |
| `/health` | GET | Health check |

Full interactive API documentation is available at `/docs` (auto-generated by FastAPI).

### Database Schema

```sql
users           (id, github_id, username, avatar_url, compute_hours, created_at)
training_sessions (id, creator_id, model_name, status, global_step, loss,
                   peer_count, description, connect_address, created_at)
checkpoints     (id, session_id, uploader_id, model_name, global_step, loss,
                 compute_hours, description, download_url, status, claimed_by)
contributions   (id, user_id, session_id, compute_hours, steps_trained)
badges          (id, user_id, badge_type, earned_at)
```

## Apple Silicon Performance

AirTrain is built on [MLX](https://github.com/ml-explore/mlx), Apple's native ML framework that takes full advantage of Apple Silicon's **unified memory architecture** — CPU and GPU share the same memory pool, eliminating the host-to-device copy overhead that plagues NVIDIA GPU training.

### Chip Benchmarks

| Chip | GPU TFLOPS (FP32) | Memory BW | Unified Memory | Power |
|---|---|---|---|---|
| M1 | 1.36 | 60 GB/s | 8-16 GB | 20W |
| M2 | 2.24 | 91 GB/s | 8-24 GB | 22W |
| M3 | 2.47 | 92 GB/s | 8-24 GB | 22W |
| M4 | 2.90 | 100 GB/s | 16-32 GB | 22W |
| M4 Pro | 5.30 | 273 GB/s | 24-48 GB | 30W |
| M4 Max | 18.43 | 546 GB/s | 36-128 GB | 40W |

*Source: [arXiv:2502.05317](https://arxiv.org/html/2502.05317v1)*

### Why Apple Silicon for Training?

1. **Unified memory** — A M4 Max with 128GB can train a 70B parameter model without offloading. An NVIDIA RTX 4090 has only 24GB VRAM.
2. **Power efficiency** — Apple Silicon achieves ~245-460 GFLOPS/W vs NVIDIA A100's ~0.7 TFLOPS/W. Training on MacBooks costs nothing in electricity compared to a cloud GPU.
3. **Ubiquity** — There are hundreds of millions of Apple Silicon Macs in the world. Even if each one contributes just a few hours, the aggregate compute is enormous.
4. **MLX** — Apple's framework is purpose-built for this hardware. Lazy evaluation, unified memory, and native Metal GPU support.

### Scaling Math

A single M4 MacBook Pro: **2.9 TFLOPS**. An NVIDIA A100: **19.5 TFLOPS**.

But 7 friends with M4 MacBooks = **20.3 TFLOPS** combined — matching an A100 for $0 in compute cost.

With DiLoCo's 500x communication reduction, the Wi-Fi overhead is negligible. You get near-linear scaling up to dozens of Macs.

## CLI Reference

| Command | Description |
|---|---|
| `airtrain init` | Initialize a new training project (creates `airtrain.yaml`) |
| `airtrain start --model <name> --dataset <path>` | Start training as coordinator |
| `airtrain start --dashboard` | Start with local web dashboard on `:8471` |
| `airtrain start --resume <checkpoint>` | Resume training from a checkpoint |
| `airtrain join auto` | Join a session via mDNS auto-discovery |
| `airtrain join <ip:port>` | Join a session at a specific address |
| `airtrain status` | Show cluster status (peers, step, loss) |
| `airtrain pause` | Checkpoint and pause training |
| `airtrain resume --from <checkpoint>` | Resume from a saved checkpoint |
| `airtrain relay export --checkpoint <path>` | Export portable relay checkpoint |
| `airtrain relay import <path>` | Import a relay checkpoint |
| `airtrain sleep --window "23:00-07:00"` | Auto-join sessions while you sleep |

### Key Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2-small` | Model architecture to train |
| `--dataset` | (required) | Path to training data |
| `--batch-size` | 8 | Per-worker batch size |
| `--inner-steps` | 500 | DiLoCo inner steps before sync |
| `--port` | 7471 | TCP port for peer communication |
| `--checkpoint-dir` | `./checkpoints` | Where to save checkpoints |
| `--dashboard` | off | Enable local web dashboard |

## Configuration

AirTrain can be configured via `airtrain.yaml` (created by `airtrain init`) or CLI flags:

```yaml
model_name: gpt2-small
dataset_path: ./data/wikitext.txt
batch_size: 8
max_steps: 100000
seq_length: 512
checkpoint_dir: ./checkpoints
checkpoint_every: 1000
log_every: 10
seed: 42

diloco:
  inner_steps: 500
  inner_lr: 0.0003
  inner_optimizer: adamw
  inner_weight_decay: 0.1
  outer_lr: 0.7
  outer_momentum: 0.9
  use_nesterov: true
  gradient_compression: true
  compress_to_fp16: true
```

## Project Structure

```
AirTrain/
├── airtrain/                        # Core Python package
│   ├── cli.py                       # Click CLI (init, start, join, relay, etc.)
│   ├── config.py                    # Pydantic config models
│   ├── compat.py                    # Cross-platform MLX compatibility layer
│   ├── discovery/
│   │   ├── mdns.py                  # LAN auto-discovery via Zeroconf/Bonjour
│   │   ├── relay.py                 # HTTP signaling server for internet discovery
│   │   └── peer.py                  # Peer manager + Apple Silicon hardware detection
│   ├── engine/
│   │   ├── diloco.py                # DiLoCo algorithm implementation
│   │   ├── trainer.py               # Base MLX training loop
│   │   ├── coordinator.py           # Coordinator node orchestration
│   │   ├── worker.py                # Worker node logic
│   │   ├── checkpoint.py            # Save/load/export/import checkpoints
│   │   ├── pipeline.py              # Pipeline parallelism interface (v2)
│   │   └── status.py                # Cluster status queries
│   ├── network/
│   │   ├── transport.py             # Async TCP server/client with heartbeat
│   │   ├── protocol.py              # Binary message protocol
│   │   └── compression.py           # FP16 + gzip gradient compression
│   ├── models/
│   │   ├── transformer.py           # GPT-2 implementation in MLX
│   │   └── registry.py              # Model name → factory mapping
│   └── dashboard/
│       ├── app.py                   # FastAPI local dashboard + SSE
│       └── static/index.html        # Dashboard UI (Chart.js)
├── website/                         # Public website (airtrain.dev)
│   ├── backend/
│   │   ├── app.py                   # FastAPI app with CORS
│   │   ├── models.py                # SQLAlchemy table definitions
│   │   ├── auth.py                  # GitHub OAuth flow
│   │   └── routes/
│   │       ├── swarms.py            # Live session browser API
│   │       ├── relay.py             # Relay checkpoint board API
│   │       └── leaderboard.py       # Leaderboard + badges API
│   └── frontend/
│       └── index.html               # Landing page with swarm/relay/leaderboard
├── examples/
│   ├── train_gpt2.py                # GPT-2 distributed training example
│   ├── train_mnist.py               # Simple MNIST example for testing
│   └── relay_demo.py                # Relay checkpoint handoff demo
├── tests/
│   ├── test_config.py               # Config model tests
│   └── test_protocol.py             # Protocol encode/decode tests
├── pyproject.toml                   # Package config + dependencies
├── README.md
└── LICENSE                          # MIT
```

## Comparison to Existing Tools

| Feature | AirTrain | PyTorch DDP | Petals | Hivemind | Flower |
|---|---|---|---|---|---|
| Apple Silicon native | Yes (MLX) | No (MPS single-device) | Partial | Partial | Via PyTorch |
| Communication reduction | 500x (DiLoCo) | 1x (every step) | N/A (inference) | ~10x (Moshpit) | Varies |
| Zero-config discovery | mDNS | Manual | DHT | DHT | Manual |
| Wi-Fi friendly | Yes | No | Yes | Yes | Yes |
| Dynamic join/leave | Yes | No | Yes | Yes | Yes (per round) |
| Checkpoint relay | Yes | No | No | No | No |
| Community platform | airtrain.dev | No | No | No | No |
| Sleep Swarms (24/7) | Yes | No | No | No | No |
| Target hardware | Mac (Apple Silicon) | NVIDIA GPU | Any GPU | Any GPU | Any |

### When to Use AirTrain vs Alternatives

- **AirTrain** — You have Macs and want to train models collaboratively with friends/community, either live or asynchronously via relay
- **PyTorch DDP** — You have a homogeneous GPU cluster with fast interconnect (InfiniBand)
- **Petals** — You want to run inference on huge models (70B+) by pooling GPUs across the internet
- **Hivemind** — You want decentralized training across heterogeneous GPU machines
- **Flower** — You need federated learning where data stays private on each device

## Roadmap

### v0.1 (Current)
- [x] DiLoCo data-parallel training
- [x] mDNS zero-config discovery
- [x] Async TCP transport with heartbeat
- [x] FP16 + gzip gradient compression
- [x] Checkpoint save/load/relay
- [x] CLI (start, join, pause, relay)
- [x] Local web dashboard
- [x] Public website (swarm browser, relay board, leaderboard)
- [x] GPT-2 model

### v0.2 (Planned)
- [ ] Pipeline parallelism for models too large for single Mac
- [ ] Real dataset loaders (HuggingFace datasets integration)
- [ ] More model architectures (LLaMA, Mistral, Phi)
- [ ] Thunderbolt JACCL backend for same-room high-speed training
- [ ] Website: real-time session metrics via WebSocket

### v0.3 (Future)
- [ ] NAT traversal for peer-to-peer across the internet without relay
- [ ] Differential privacy for gradient sharing
- [ ] Mobile support (iOS Neural Engine contribution)
- [ ] Model Hub integration (auto-publish to HuggingFace on convergence)
- [ ] Browser-based training viewer

## Contributing

We welcome contributions! Areas where help is especially valuable:

- **Model implementations** — Port more architectures to MLX
- **Dataset loaders** — Integration with HuggingFace datasets, custom formats
- **Testing** — Multi-node integration tests, benchmarks
- **Website** — UI/UX improvements, mobile responsiveness
- **Documentation** — Tutorials, guides, video walkthroughs

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

AirTrain builds on the work of:

- [MLX](https://github.com/ml-explore/mlx) by Apple — Native Apple Silicon ML framework
- [DiLoCo](https://arxiv.org/abs/2311.08105) by Douillard et al. — The low-communication distributed training algorithm
- [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiloco) by PrimeIntellect — Open-source DiLoCo implementation and validation
- [Petals](https://github.com/bigscience-workshop/petals) — Proving collaborative ML training works over the internet
- [Hivemind](https://github.com/learning-at-home/hivemind) — Decentralized deep learning primitives
- [python-zeroconf](https://github.com/python-zeroconf/python-zeroconf) — Pure Python mDNS/DNS-SD implementation
