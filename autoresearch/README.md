# Autoresearch on the HF Cluster

Run [karpathy/autoresearch](https://github.com/karpathy/autoresearch) overnight on H100s with ml-intern driving the experiment loop.

## One-line summary

Agent edits `train.py`, runs 12-min training experiments, keeps/discards based on
`val_bpb`, commits to git, and repeats until wall clock expires.

---

## Directory layout on cluster

```
/fsx/aksel/autoresearch/
├── slurm/
│   ├── run.sbatch            # Main SLURM job — launches the agent
│   └── guided_prompt.txt     # Optional: community findings + research loop
├── cache/                    # Shared data cache (downloaded ONCE)
│   ├── data/                 # ~6 GB of ClimbMix parquet shards
│   └── tokenizer/            # BPE tokenizer trained on the shards
├── logs/                     # SLURM stdout/stderr
│   ├── ar_<jobid>.out
│   └── ar_<jobid>.err
└── runs/
    └── run_<jobid>/
        ├── prompt.txt        # What the agent received
        ├── timer.sh          # Wall-clock timer for --resume reprompting
        ├── agent_out.txt     # Raw agent stdout
        ├── time_taken.txt    # HH:MM:SS
        └── repo/             # autoresearch clone — agent commits go here
            ├── train.py      # ← the agent modifies this
            ├── results.tsv   # commit hash → val_bpb → keep/discard
            ├── run.log       # last `uv run train.py` output
            └── session_logs/ # full agent trace (events[] + messages[])
```

---

## Deploy / update

The sbatch is versioned at `autoresearch/slurm/run.sbatch` in this repo and must
be mirrored to `/fsx/aksel/autoresearch/slurm/run.sbatch` on the cluster:

```bash
# From this repo root:
scp autoresearch/slurm/run.sbatch hpc-cluster-hopper-login-node-1:/fsx/aksel/autoresearch/slurm/run.sbatch
```

---

## Submit a run

```bash
# From the cluster login node:
cd /fsx/aksel/autoresearch
export PATH=/opt/slurm/bin:$PATH
set -a && source /fsx/aksel/hf_agent/.env && set +a   # loads ANTHROPIC_API_KEY etc.

# Vanilla: agent reads program.md and goes
sbatch slurm/run.sbatch 8 claude-opus-4-6

# Guided: appends community findings (batch-halving, width scaling, QK-Norm fix, etc.)
sbatch slurm/run.sbatch 8 claude-opus-4-6 guided

# Resume a prior run (repo, branch, and results.tsv must exist)
sbatch slurm/run.sbatch 4 claude-opus-4-6 guided 22070536
```

**Arguments:** `[hours] [agent_config] [mode] [resume_run_id]`
- `hours` — time budget for the agent (default 8). The `timer.sh` reports this.
- `agent_config` — short name, prefixed with `anthropic/` at runtime (default `claude-opus-4-6`).
- `mode` — `vanilla` or `guided`.
- `resume_run_id` — optional SLURM job id of a prior run to continue from.

**SLURM wall time is 16h.** Agent `hours` should be ≤ wall time minus ~20 min
startup/container-pull margin.

---

## Monitoring

```bash
# Running jobs
export PATH=/opt/slurm/bin:$PATH
squeue -u aksel

# Live agent output
tail -f /fsx/aksel/autoresearch/runs/run_<jobid>/agent_out.txt

# Experiment results so far
cat /fsx/aksel/autoresearch/runs/run_<jobid>/repo/results.tsv

# Git history of experiments
cd /fsx/aksel/autoresearch/runs/run_<jobid>/repo && git log --oneline
```

---

## How it works

1. **Host side (before srun):**
   - Clone `karpathy/autoresearch` into `runs/run_<jobid>/repo/` (unless resuming).
   - Read `program.md` to build the agent prompt. Append `guided_prompt.txt` if `mode=guided`.
   - Generate `timer.sh` that reports `(JOB_END_EPOCH - now)` as `H:MM`.

2. **Container side (srun inside posttrainbench:latest):**
   - Clone autoresearch if missing (safety net).
   - `uv sync` — creates `.venv/` inside repo with pinned torch 2.9.1.
     Reused across runs because `.venv/` persists on `/fsx`.
   - `uv run prepare.py` — downloads data shards + trains BPE tokenizer.
     Fast no-op after the first run thanks to the shared cache mount.
   - `git checkout -b autoresearch/run_<jobid>` for a fresh experiment branch.
   - Copy `timer.sh` into the repo so the agent can call `bash timer.sh`.
   - Install ml-intern from the `posttrain-bench` branch into system Python.
   - Launch: `python -m agent.main --max-iterations -1 --no-stream --resume timer.sh --resume-min-minutes 15 --model anthropic/<cfg> "<prompt>"`.

3. **Agent loop:**
   - Reads `program.md` (the autoresearch "skill"), understands the loop.
   - Edits `train.py`, runs `uv run train.py > run.log 2>&1` (~12 min).
   - Extracts `val_bpb`, records in `results.tsv`, commits.
   - If it exits early (`turn_complete`), the `--resume` loop re-prompts as
     long as `timer.sh` reports ≥15 min remaining.
   - Keeps going until the timer expires → one final turn, then exits.

---

## Container requirements

Uses `registry.hpc-cluster-hopper.hpc.internal.huggingface.tech/library/posttrainbench:latest`.

Autoresearch needs Python 3.11 + CUDA + uv + git — all present. Autoresearch's
own pinned `torch==2.9.1` goes into its `.venv/` via `uv sync`, isolated from
the container's system Python which is used by ml-intern.

---

## Gotchas

- **Data cache path matters.** The shared mount `/fsx/aksel/autoresearch/cache`
  → `/root/.cache/autoresearch` means `prepare.py` is a no-op after the first
  run. If you delete the cache, the next run downloads ~6 GB again.

- **`.venv/` on /fsx is slow but persistent.** First `uv sync` takes ~2 min;
  subsequent runs use cached wheels. Don't delete `runs/run_<jobid>/repo/.venv/`
  unless dependencies changed.

- **Agent writes `session_logs/` into the repo.** Git won't commit these (the
  agent only commits `train.py` changes per program.md), but they take disk.
  Each session json has the full `events[]` trace which survives compaction.

- **Rate limits.** If Anthropic rate-limits during a 16h run, the agent errors
  out and the SLURM job ends early. Resume with
  `sbatch slurm/run.sbatch <hours_left> <cfg> <mode> <prior_jobid>`.

- **Timer granularity.** `timer.sh` rounds down to the minute. The reprompt
  checks `--resume-min-minutes 15`, so the agent gets reprompted up until
  there's <15 min left; it then has those 15 min for a final turn to wind down.

- **`set -euo pipefail` in the container `bash -c` means unset env vars blow
  up.** All env refs use `${VAR:-}` fallback. Keep it that way when editing.
