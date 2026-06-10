# GW Merger Bench

A benchmark for evaluating AI agents on gravitational-wave parameter estimation. Agents analyse synthetic binary black hole (BBH) strain data from LIGO-style detectors and recover physical parameters. The benchmark grades the final submission on a conjunction gate that separates statistical fit quality from physical correctness.

---

## What Each Task Gives the Agent

Each task provides:

- A 16-second strain time series from two detectors (H1 and L1) as `.npy` files
- The detector noise PSD as `.npy` files
- A `task.json` with physics metadata (sample rate, f_lower, approximant, segment duration)

---

## Evaluation Criteria

A task **passes only if all four criteria pass simultaneously** (conjunction gate):

| Criterion | What it checks | Threshold |
|---|---|---|
| `ok_waveform_match` | Noise-weighted overlap between reconstructed waveform and observed strain | ≥ 0.90 |
| `ok_chirp_mass` | Chirp mass error vs true value | ≤ 5% |
| `ok_mass_ratio` | Mass ratio absolute error vs true value | ≤ 0.15 |
| `ok_merger_type` | BBH / BNS / NSBH exact string match | exact |

### How `ok_waveform_match` works

The evaluator computes this entirely independently — the agent cannot self-report it:

```
Agent submits: mass1=32, mass2=24, spin1z=0.1, distance=450, ra=1.2, dec=-0.5 ...
                        ↓
Evaluator reads approximant from ground_truth.json (e.g. IMRPhenomD)
                        ↓
Runs that approximant with submitted parameters → clean template h(t)
                        ↓
Projects onto H1 detector using submitted sky location
                        ↓
Tries 4 coalescence phases (0, π/2, π, 3π/2) — takes best overlap
                        ↓
Noise-weighted overlap: <h_template | strain_H1> / sqrt(<h|h> × <s|s>)
                        ↓
Passes if overlap ≥ 0.90
```

The gap between `ok_waveform_match` passing and `ok_chirp_mass` failing is the core **"good statistics ≠ good physics"** signal — the agent found a waveform that fits the data but with wrong physical parameters. This is tracked as `stat_pass_phys_fail` in the results.

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
│   └── evaluator.py          — conjunction gate + waveform overlap forward model
│
├── data/
│   ├── IMRPhenomD/           ← approximant subfolder
│   │   ├── index.json
│   │   ├── 000/
│   │   │   ├── strain_H1.npy
│   │   │   ├── strain_L1.npy
│   │   │   ├── psd_H1.npy
│   │   │   ├── psd_L1.npy
│   │   │   ├── psd_freqs.npy
│   │   │   ├── times.npy
│   │   │   ├── task.json           — public (no tier/difficulty)
│   │   │   └── ground_truth.json   — hidden (tier, difficulty, true params)
│   │   ├── 001/
│   │   └── ...
│   ├── SEOBNRv4/             ← separate subfolder per approximant
│   └── IMRPhenomXHM/
│
└── results/
    └── easy_2026-06-10_12-00-00/
        ├── run_summary.json
        ├── 000.json
        └── ...
```

All tasks are flat within each approximant subfolder — no `easy/medium/hard` subfolders. Tier is stored only in `ground_truth.json`.

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
# Default: IMRPhenomD, saved to data/IMRPhenomD/
python scripts/generate_dataset.py --seed 42

# Explicit approximant
python scripts/generate_dataset.py --seed 42 --approximant SEOBNRv4

# Custom base output dir
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD --outdir data
```

Generates **300 tasks** (100 easy / 100 medium / 100 hard) saved to `data/{approximant}/`. Takes 5–10 minutes.

### Options

| Argument | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--approximant` | `IMRPhenomD` | Waveform model — also becomes the subfolder name |
| `--outdir` | `data` | Base output directory |

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

No `tier`, no `difficulty_score`.

### What ground_truth.json contains (hidden from agent)

```json
{
    "task_id":          "000",
    "tier":             "easy",
    "difficulty_score": 2,
    "chirp_mass":       28.04,
    "mass1":            32.1,
    "mass2":            22.8,
    "mass_ratio":       0.71,
    "spin1z":           0.05,
    "spin2z":          -0.02,
    "distance":         450.0,
    "network_snr":      24.3,
    "merger_type":      "BBH",
    "approximant":      "IMRPhenomD",
    ...
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

### What input.json contains (what your pipeline receives)

```json
{
    "task_id":            "000",
    "task_description":   "...",
    "approximant":        "IMRPhenomD",
    "sample_rate_hz":     2048,
    "segment_duration_s": 16,
    "f_lower_hz":         20.0,
    "data_paths": {
        "strain_H1":  "/absolute/path/strain_H1.npy",
        "strain_L1":  "/absolute/path/strain_L1.npy",
        "psd_H1":     "/absolute/path/psd_H1.npy",
        "psd_L1":     "/absolute/path/psd_L1.npy",
        "psd_freqs":  "/absolute/path/psd_freqs.npy",
        "times":      "/absolute/path/times.npy"
    },
    "submission_format":  { ... },
    "output_path":        "/tmp/xxx/output.json"
}
```

### What output.json must contain

All 13 fields required:

```json
{
    "chirp_mass_Msun": 28.5,
    "mass1_Msun":      32.0,
    "mass2_Msun":      24.0,
    "mass_ratio":      0.75,
    "spin1z":          0.1,
    "spin2z":          0.05,
    "distance_Mpc":    450.0,
    "inclination_rad": 0.4,
    "ra_rad":          1.2,
    "dec_rad":         -0.5,
    "network_snr":     20.0,
    "merger_type":     "BBH",
    "confidence":      0.8
}
```

### Run commands

```bash
# 1 task — quick test
python scripts/run_benchmark.py \
    --pipeline-path /path/to/your/pipeline \
    --pipeline-entry run_gw_benchmark.py \
    --data-dir data/IMRPhenomD \
    --tier easy --max-tasks 1 --verbose

# Full easy tier
python scripts/run_benchmark.py \
    --pipeline-path /path/to/your/pipeline \
    --pipeline-entry run_gw_benchmark.py \
    --data-dir data/IMRPhenomD \
    --tier easy \
    --outfile results/my_pipeline_easy.json

# All tiers
python scripts/run_benchmark.py \
    --pipeline-path /path/to/your/pipeline \
    --pipeline-entry run_gw_benchmark.py \
    --data-dir data/IMRPhenomD \
    --tier all \
    --outfile results/my_pipeline_full.json

# Different approximant
python scripts/run_benchmark.py \
    --pipeline-path /path/to/your/pipeline \
    --pipeline-entry run_gw_benchmark.py \
    --data-dir data/SEOBNRv4 \
    --tier easy
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--pipeline-path` | required | Absolute path to your pipeline repo root |
| `--pipeline-entry` | `run_gw_benchmark.py` | Entry point script relative to `--pipeline-path` |
| `--pipeline-timeout` | `300` | Seconds before pipeline is killed per task |
| `--tier` | `all` | `easy`, `medium`, `hard`, or `all` |
| `--max-tasks` | None | Limit tasks — useful for quick testing |
| `--data-dir` | `data/IMRPhenomD` | Path to approximant subfolder |
| `--outfile` | None | Also save full report to this path |
| `--verbose` | False | Print pipeline stdout and submission details |

---

## Output Format

### Live output per task

```
[001/300] 000        tier=easy   PASS  crit=4/4  t=18.4s
[002/300] 001        tier=easy   FAIL  crit=2/4  t=21.1s
```

### Per-task JSON (saved immediately after each task)

```json
{
  "task_id":    "000",
  "tier":       "easy",
  "passed":     true,
  "elapsed_s":  18.4,
  "submission": { ... },
  "metrics": {
    "passed":                true,
    "n_criteria_passed":     4,
    "ok_waveform_match":     true,
    "ok_chirp_mass":         true,
    "ok_mass_ratio":         true,
    "ok_merger_type":        true,
    "waveform_overlap":      0.923,
    "chirp_mass_submitted":  28.5,
    "chirp_mass_true":       28.04,
    "chirp_mass_frac_err":   0.016,
    "mass_ratio_submitted":  0.74,
    "mass_ratio_true":       0.71,
    "mass_ratio_abs_err":    0.03,
    "merger_type_submitted": "BBH",
    "merger_type_true":      "BBH",
    "stat_pass_phys_fail":   false
  }
}
```

### Summary table

```
Tier       Pass              Mc err%    q err    Overlap    Stat✓Phys✗
easy       70/100 (70%)       4.32%    0.091      0.941        12%
medium     35/100 (35%)      14.21%    0.162      0.823        28%
hard       10/100 (10%)      38.74%    0.291      0.601        41%
overall   115/300 (38%)      19.09%    0.181      0.788        27%
```

| Column | Description |
|---|---|
| `Pass` | Tasks where all four criteria passed simultaneously |
| `Mc err%` | Mean chirp mass percentage error |
| `q err` | Mean mass ratio absolute error |
| `Overlap` | Mean noise-weighted waveform overlap |
| `Stat✓Phys✗` | Waveform matched (≥ 0.90) but chirp mass failed — the "good statistics ≠ good physics" gap |

---

## Difficulty Tiers

Tier is stored in `ground_truth.json` only — the agent never sees it:

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| `network_snr_range` | 20–35 | 12–20 | 8–12 |
| `total_mass_range` (M☉) | 40–80 | 25–120 | 10–200 |
| `mass_ratio_range` | 0.7–1.0 | 0.4–0.9 | 0.1–0.6 |
| `spin_magnitude_range` | 0–0.1 | 0–0.5 | 0.3–0.9 |
| `inclination_range` (rad) | 0–0.3 | 0–1.0 | 0.5–π/2 |

---

## Generating Multiple Approximant Datasets

Use the same `--seed` to keep physical parameters identical — only the waveform physics changes:

```bash
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD
python scripts/generate_dataset.py --seed 42 --approximant SEOBNRv4
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomXHM
```

Each generates `data/IMRPhenomD/`, `data/SEOBNRv4/`, `data/IMRPhenomXHM/` with the same physical events but different waveform templates — useful for testing agent robustness across approximants.
