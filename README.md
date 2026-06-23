# GW Merger Bench

A benchmark for evaluating AI agents on gravitational-wave (GW) parameter estimation. Agents analyse synthetic binary black hole (BBH) strain data from LIGO-style detectors and recover physical parameters. The benchmark grades submissions on a conjunction of physics-grounded criteria, separating statistical fit quality from physical correctness.

---

## What the Agent Submits

| Parameter | Description |
|---|---|
| `chirp_mass_Msun` | Chirp mass in solar masses |
| `mass1_Msun` | Primary component mass |
| `mass2_Msun` | Secondary component mass |
| `mass_ratio` | q = m2/m1, range (0, 1] |
| `network_snr` | Estimated SNR from matched filter |
| `merger_type` | Exactly `"BBH"`, `"BNS"`, or `"NSBH"` |

Parameters like spins, distance, sky location, and inclination are **not evaluated** — they cannot be reliably recovered from single-detector matched filtering.

---

## What Each Task Gives the Agent

- A 16-second strain time series from H1 and L1 as `.npy` files
- The detector noise PSD as `.npy` files
- A `task.json` with physics metadata (sample rate, f_lower, approximant hint, segment duration)

**Never given to the agent:** difficulty tier, true parameter values, feedback.

---

## Evaluation Criteria

A task **passes only if all three criteria pass simultaneously**:

| Criterion | What it checks | Threshold |
|---|---|---|
| `ok_chirp_mass` | Chirp mass fractional error vs true value | ≤ tier-dependent (5/8/10%) |
| `ok_mass_ratio` | Mass ratio absolute error vs true value | ≤ tier-dependent (0.15/0.20/0.25) |
| `ok_merger_type` | BBH / BNS / NSBH exact string match | exact |

Waveform overlap (`ok_waveform_match`) is **computed and reported as a diagnostic** but does not gate pass/fail. It depends on extrinsic parameters (sky, distance, inclination) the agent cannot recover, so including it in the pass gate would unfairly penalise the agent for parameters it was never designed to estimate.

### The stat_pass_phys_fail diagnostic

`stat_pass_phys_fail = True` when `ok_waveform_match=True` but `ok_chirp_mass=False` — the agent found a waveform that fits the data but at the wrong physical parameters. This is the core **"good statistics ≠ good physics"** signal.

---

## Difficulty Tiers

Tier is stored in `ground_truth.json` only — the agent never sees it:

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| `network_snr_range` | 20–35 | 12–20 | 8–12 |
| `total_mass_range` (M☉) | 40–80 | 25–120 | 10–200 |
| `mass_ratio_range` | 0.7–1.0 | 0.4–0.9 | 0.1–0.6 |
| `spin_magnitude_range` | 0.0–0.0 | 0.0–0.0 | 0.0–0.0 |
| `inclination_range` (rad) | 0.0–0.0 | 0.0–0.0 | 0.0–0.0 |
| `chirp_mass_tol_frac` | 0.05 | 0.08 | 0.10 |
| `mass_ratio_tol_abs` | 0.15 | 0.20 | 0.25 |

Current dataset (`IMRPhenomD_zerospin`) uses zero spin and face-on inclination for all tiers — a controlled benchmark isolating Mc/q recovery as the sole difficulty axis. Difficulty increases via lower SNR and more unequal mass ratios.

---

## Repository Structure

```
GW_merger_bench/
│
├── scripts/
│   ├── generate_dataset.py   — generates synthetic BBH tasks
│   └── run_benchmark.py      — runs any external pipeline, saves results
│
├── evaluation/
│   └── evaluator.py          — 3-criterion conjunction gate + waveform overlap
│
├── data/
│   ├── IMRPhenomD/           ← original dataset (with spins, old tolerances)
│   ├── IMRPhenomD_zerospin/  ← current benchmark dataset
│   │   └── IMRPhenomD/
│   │       ├── index.json
│   │       ├── 000/
│   │       │   ├── strain_H1.npy
│   │       │   ├── strain_L1.npy
│   │       │   ├── psd_H1.npy
│   │       │   ├── psd_L1.npy
│   │       │   ├── psd_freqs.npy
│   │       │   ├── task.json           — public (no tier/difficulty)
│   │       │   └── ground_truth.json   — hidden (tier, true params, tolerances)
│   │       └── ...
│   ├── SEOBNRv4/
│   └── IMRPhenomXHM/
│
└── results/
    ├── plots/                    — chirp signal plots per task
    ├── agent_logs/               — full agent step logs per task
    └── all_2026-06-22_16-39-22/
        ├── run_summary.json
        ├── 000.json
        └── ...
```

---

## Installation

```bash
cd ~/Desktop/code/GW_merger_bench

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install pycbc numpy scipy h5py
```

Verify:

```bash
python -c "import pycbc; print('pycbc', pycbc.__version__)"
python -c "from pycbc.waveform import get_td_waveform; print('waveform ok')"
```

---

## Generating the Dataset

```bash
# Zero-spin controlled benchmark (current default)
python scripts/generate_dataset.py \
    --seed 42 \
    --approximant IMRPhenomD \
    --outdir data/IMRPhenomD_zerospin

# Multiple approximants with same physical parameters (same seed)
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD
python scripts/generate_dataset.py --seed 42 --approximant SEOBNRv4
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomXHM
```

Generates **15 tasks** (5 easy / 5 medium / 5 hard) by default. Edit `DIFFICULTY_CONFIG` in `generate_dataset.py` to change `n_tasks` per tier.

### What task.json contains (given to agent)

```json
{
    "task_id":          "000",
    "description":      "A gravitational-wave strain signal has been recorded...",
    "sample_rate":      2048,
    "segment_duration": 16,
    "f_lower":          20.0,
    "detectors":        ["H1", "L1"],
    "approximant_hint": "IMRPhenomD",
    "submission_format": { ... }
}
```

No `tier`, no `difficulty_score`, no true parameters.

### What ground_truth.json contains (hidden from agent)

```json
{
    "task_id":                  "000",
    "tier":                     "easy",
    "difficulty_score":         2,
    "chirp_mass":               28.04,
    "mass1":                    32.1,
    "mass2":                    22.8,
    "mass_ratio":               0.71,
    "spin1z":                   0.0,
    "spin2z":                   0.0,
    "distance":                 450.0,
    "inclination":              0.0,
    "coalescence_time":         10.72,
    "network_snr":              24.3,
    "merger_type":              "BBH",
    "approximant":              "IMRPhenomD",
    "chirp_mass_tol_frac":      0.05,
    "mass_ratio_tol_abs":       0.15,
    "snr_tol_frac":             0.20
}
```

---

## Running the Benchmark

### How it works

```
run_benchmark.py
      ↓
reads task.json → writes input.json → calls your pipeline → reads output.json
      ↓
evaluator.py scores output.json against ground_truth.json
      ↓
saves per-task JSON + run_summary.json
```

### What input.json contains

```json
{
    "task_id":            "000",
    "approximant":        "IMRPhenomD",
    "sample_rate_hz":     2048,
    "f_lower_hz":         20.0,
    "data_paths": {
        "strain_H1":  "/absolute/path/strain_H1.npy",
        "strain_L1":  "/absolute/path/strain_L1.npy",
        "psd_H1":     "/absolute/path/psd_H1.npy",
        "psd_L1":     "/absolute/path/psd_L1.npy",
        "psd_freqs":  "/absolute/path/psd_freqs.npy"
    },
    "output_path": "/tmp/xxx/output.json"
}
```

### What output.json must contain

```json
{
    "chirp_mass_Msun": 28.5,
    "mass1_Msun":      32.0,
    "mass2_Msun":      24.0,
    "mass_ratio":      0.75,
    "network_snr":     20.0,
    "merger_type":     "BBH"
}
```

Missing fields are filled with safe defaults. If the pipeline crashes or times out, a blank submission is recorded and the benchmark continues.

### Run commands

```bash
# Single task quick test
python scripts/run_benchmark.py \
    --pipeline-path /home/sr/Desktop/code/physics_agent_harness \
    --pipeline-entry run.py \
    --data-dir data/IMRPhenomD_zerospin/IMRPhenomD \
    --tier easy --max-tasks 1 \
    --pipeline-timeout 1800 \
    --verbose

# Full benchmark — all 15 tasks
python scripts/run_benchmark.py \
    --pipeline-path /home/sr/Desktop/code/physics_agent_harness \
    --pipeline-entry run.py \
    --data-dir data/IMRPhenomD_zerospin/IMRPhenomD \
    --tier all \
    --pipeline-timeout 1800 \
    --outfile results/full_zerospin_run.json \
    --verbose

# Background run with log
nohup python scripts/run_benchmark.py \
    --pipeline-path /home/sr/Desktop/code/physics_agent_harness \
    --pipeline-entry run.py \
    --data-dir data/IMRPhenomD_zerospin/IMRPhenomD \
    --tier all \
    --pipeline-timeout 1800 \
    --outfile results/full_zerospin_run.json \
    > results/run_log.txt 2>&1 &
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--pipeline-path` | required | Absolute path to pipeline repo root |
| `--pipeline-entry` | `run.py` | Entry point script |
| `--pipeline-timeout` | `300` | Seconds before pipeline killed per task (use 1800 for PE) |
| `--tier` | `all` | `easy`, `medium`, `hard`, or `all` |
| `--max-tasks` | None | Limit tasks — useful for quick testing |
| `--data-dir` | `data/IMRPhenomD` | Path to dataset (index.json must exist here) |
| `--outfile` | None | Also save full report to this path |
| `--verbose` | False | Print pipeline stdout |

---

## Output Format

### Live output per task

```
[001/15] 000        tier=easy   PASS  crit=3/3  t=471s
[002/15] 001        tier=easy   FAIL  crit=2/3  t=505s
```

### Summary table

```
Tier       Pass            Mc err%   q err   Overlap  Stat✓Phys✗
--------------------------------------------------------------
easy       1/5 (20%)        61.1%   0.322     0.356          0%
medium     2/5 (40%)        24.6%   0.149     0.710          0%
hard       1/5 (20%)        20.3%   0.222     0.693         40%
overall    4/15 (27%)       35.4%   0.231     0.586         13%
```

| Column | Description |
|---|---|
| `Pass` | Tasks where all three criteria passed |
| `Mc err%` | Mean chirp mass percentage error |
| `q err` | Mean mass ratio absolute error |
| `Overlap` | Mean noise-weighted waveform overlap (diagnostic only) |
| `Stat✓Phys✗` | Waveform matched but chirp mass failed — the "good statistics ≠ good physics" signal |

---

## Known Failure Modes

| Failure type | Cause | Fix |
|---|---|---|
| Timeout (blank submission) | PE too slow for slice sampler | Revert to `rwalk`/`walks=50`, or increase `--pipeline-timeout` |
| q bias toward equal mass | Likelihood nearly flat in q at low SNR | Higher `nlive` (500+), or wider `mass_ratio_tol_abs` |
| stat✓phys✗ on hard tasks | Sampler finds local likelihood maximum | Wider Mc prior window, higher `nlive` |
| Mc error >20% on high-mass hard tasks | Signal near `f_lower=20Hz` boundary | Raise `f_lower` detection, or lower ISCO frequency check |

---

## Evaluation Thresholds

Thresholds are baked into `ground_truth.json` at generation time:

```python
# generate_dataset.py
chirp_mass_tol_frac = {"easy": 0.05, "medium": 0.08, "hard": 0.10}[tier]
mass_ratio_tol_abs  = {"easy": 0.15, "medium": 0.20, "hard": 0.25}[tier]
```

`OVERLAP_THRESHOLD = 0.80` in `evaluator.py` can be changed without regenerating data — it's only used for the diagnostic `ok_waveform_match` flag.