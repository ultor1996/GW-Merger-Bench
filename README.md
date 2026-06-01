# GW Merger Bench

An interactive benchmark environment for evaluating AI agents on gravitational-wave parameter estimation. Agents analyse synthetic binary black hole (BBH) strain data from LIGO-style detectors, recover physical parameters, and are graded on a conjunction gate that separates statistical fit quality from physical correctness.

---

## Overview

Each task gives an agent:
- A 16-second strain time series from two detectors (H1 and L1)
- The detector noise PSD
- Two tools: `execute_python` (stateful Python sandbox) and `submit_action` (graded submission)

The agent works in **turns**. One turn = all the analysis code the agent wants to run + exactly one submission. After submitting, the agent receives criterion-level feedback and starts the next turn.

A task **passes only if all four criteria pass simultaneously** (conjunction gate):

| Criterion | What it checks | How it is computed |
|---|---|---|
| `ok_waveform_match` | Noise-weighted overlap between reconstructed waveform and observed strain ≥ 0.90 | **Evaluator runs the waveform model** on submitted parameters — agent cannot self-report this |
| `ok_chirp_mass` | Chirp mass within 5% of true value | Compared against ground truth |
| `ok_mass_ratio` | Mass ratio within 0.15 absolute | Compared against ground truth |
| `ok_merger_type` | BBH / BNS / NSBH exact match | Compared against ground truth |

The gap between `ok_waveform_match` passing and `ok_chirp_mass` failing is the core **"good statistics ≠ good physics"** signal.

---

## How `ok_waveform_match` Works

The evaluator computes this entirely independently of what the agent claims:

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

The evaluator always uses the **same waveform model the data was generated with** — stored in each task's `ground_truth.json` and read automatically. No mismatch possible.

---

## Turn Design (Interactive Agents)

```
Turn 1:
  execute_python(load data)            ← free, no turn cost
  execute_python(whiten + spectrogram) ← free, up to --max-calls-per-turn
  execute_python(estimate chirp mass)  ← free
  submit_action(estimates)             ← ends the turn, returns feedback

Turn 2:
  execute_python(refine based on feedback)
  submit_action(revised estimates)     ← ends the turn, returns feedback

... up to --max-turns
```

`execute_python` does not count against the turn counter. Only `submit_action` ends a turn. If `--max-calls-per-turn` is reached without a submission, a blank submission is forced.

**This turn design applies only to interactive agents (test agent, custom LLM agents). External pipeline agents use single-shot evaluation — see below.**

---

## Feedback Design

Feedback is entirely **hardcoded Python** — no LLM involved. The evaluator computes pass/fail and error magnitudes arithmetically.

The agent **never sees true parameter values**. On failure it sees the error magnitude; on pass it sees only the description.

**What the agent sees after a failed submission:**

```
**submit_action feedback:**
❌ Not yet passing. 9 turn(s) remaining.

Criteria:
  ❌ ok_waveform_match: Passes if noise-weighted waveform overlap >= 0.90  →  overlap: 0.023 (need >= 0.9)
  ❌ ok_chirp_mass:     Passes if chirp mass is within 5% of true value     →  error: 78.33%
  ✅ ok_mass_ratio:     Passes if mass ratio is within 0.15 of true value
  ❌ ok_merger_type:    Passes if merger type exactly matches (BBH / BNS / NSBH)  →  you submitted: BNS

1/4 criteria passed.
Turn: 1  |  Turns remaining: 9  |  Done: False

Refine your analysis and resubmit in the next turn. You have up to 8 execute_python calls before submission is forced.
```

---

## Repository Structure

```
BBH_interactive_datastet/
│
├── scripts/
│   ├── generate_dataset.py   — generates synthetic BBH tasks
│   └── run_benchmark.py      — runs any agent on tasks, collects results
│
├── environment/
│   └── gw_environment.py     — interactive episode environment
│
├── evaluation/
│   └── evaluator.py          — conjunction gate, ok_waveform_match forward model recompute
│
├── agents/
│   ├── system_prompt.py      — system prompt given to LLM agents
│   ├── test_agent.py         — generic ReAct agent using any OpenAI-compatible endpoint
│   └── external_agent.py     — black-box adapter for external pipeline evaluation
│
├── external/
│   └── pipeline_template.py  — copy this into any external pipeline repo to evaluate it
│
├── data/
│   ├── synthetic/            — default dataset (IMRPhenomD)
│   ├── seobnrv4/             — dataset generated with SEOBNRv4 (optional)
│   └── xhm/                  — dataset generated with IMRPhenomXHM (optional)
│       ├── index.json
│       ├── easy/
│       │   └── synthetic_easy_001/
│       │       ├── strain_H1.npy
│       │       ├── strain_L1.npy
│       │       ├── psd_H1.npy
│       │       ├── psd_L1.npy
│       │       ├── psd_freqs.npy
│       │       ├── times.npy
│       │       ├── task.json           — public task description (given to agent)
│       │       └── ground_truth.json   — true parameters + approximant (hidden)
│       ├── medium/
│       └── hard/
│
└── results/
    └── test_easy_2026-05-28_15-03-41/
        ├── run_config.json
        ├── run_summary.json
        └── tasks/
            └── synthetic_easy_001/
                ├── episode_summary.json
                └── turns/
                    ├── turn_01/
                    │   ├── analysis_code.py
                    │   ├── analysis_output.txt
                    │   ├── submission.json
                    │   ├── feedback.json
                    │   └── files/
                    └── turn_02/
                        └── ...
```

---

## Installation

```bash
cd ~/Desktop/code/BBH_interactive_datastet

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install pycbc numpy scipy h5py openai

touch environment/__init__.py evaluation/__init__.py agents/__init__.py
```

Verify:
```bash
python -c "import pycbc; print('pycbc', pycbc.__version__)"
python -c "from pycbc.waveform import get_td_waveform; print('waveform ok')"
```

---

## Generating the Dataset

```bash
python scripts/generate_dataset.py --seed 42 --outdir data/synthetic
```

Generates **60 tasks** (20 easy / 20 medium / 20 hard). Takes 2–5 minutes.

### Options

| Argument | Default | Description |
|---|---|---|
| `--seed` | `42` | Random seed for reproducibility |
| `--outdir` | `data/synthetic` | Output directory |
| `--approximant` | `IMRPhenomD` | Waveform model: `IMRPhenomD`, `SEOBNRv4`, or `IMRPhenomXHM` |

### Supported approximants

| Approximant | Physics | Extra parameters needed |
|---|---|---|
| `IMRPhenomD` | Full inspiral-merger-ringdown, aligned spins | None — default |
| `SEOBNRv4` | Same parameter set, different underlying physics | None — drop-in swap |
| `IMRPhenomXHM` | Includes higher-order modes, better for unequal mass | None — drop-in swap |

### Generating multiple datasets with different approximants

Using the same `--seed` keeps physical parameters identical — only the waveform physics differs. This lets you compare agent performance across approximants on identical tasks:

```bash
python scripts/generate_dataset.py --seed 42 --outdir data/imrphenomd --approximant IMRPhenomD
python scripts/generate_dataset.py --seed 42 --outdir data/seobnrv4   --approximant SEOBNRv4
python scripts/generate_dataset.py --seed 42 --outdir data/xhm        --approximant IMRPhenomXHM
```

The chosen approximant is stored in every task's `ground_truth.json`. The evaluator reads it automatically — no manual configuration needed.

### Changing tasks per tier

```python
# scripts/generate_dataset.py — DIFFICULTY_CONFIG
"easy":   { "n_tasks": 20, ... },
"medium": { "n_tasks": 20, ... },
"hard":   { "n_tasks": 20, ... },
```

### Changing physical parameter ranges

```python
"easy": {
    "network_snr_range":      (20.0, 35.0),
    "total_mass_range":       (40.0, 80.0),
    "mass_ratio_range":       (0.7,  1.0),
    "spin_magnitude_range":   (0.0,  0.1),
    "inclination_range":      (0.0,  0.3),
    "difficulty_score_range": (1, 3),
},
```

| Parameter | Easy | Medium | Hard | Effect |
|---|---|---|---|---|
| `network_snr_range` | 20–35 | 12–20 | 8–12 | Lower SNR = harder to detect |
| `total_mass_range` | 40–80 M☉ | 25–120 M☉ | 10–200 M☉ | Extreme masses = fewer cycles in band |
| `mass_ratio_range` | 0.7–1.0 | 0.4–0.9 | 0.1–0.6 | Unequal mass = harder parameter recovery |
| `spin_magnitude_range` | 0–0.1 | 0–0.5 | 0.3–0.9 | High spin = waveform phase degeneracy |
| `inclination_range` | 0–0.3 rad | 0–1.0 rad | 0.5–π/2 rad | Edge-on systems = weaker signal |

### Changing evaluation thresholds

```python
# scripts/generate_dataset.py — inside generate_one_event()
chirp_mass_tol_frac = 0.05   # 5%  chirp mass tolerance
mass_ratio_tol_abs  = 0.15   # 0.15 absolute mass ratio tolerance
```

The waveform overlap threshold is in `evaluation/evaluator.py`:
```python
OVERLAP_THRESHOLD = 0.90   # change here — takes effect without regeneration
```

> **Important**: thresholds in `generate_dataset.py` are baked into `ground_truth.json` at generation time. Always regenerate after changing them.

```bash
python scripts/generate_dataset.py --seed 42 --outdir data/synthetic
```

---

## Running the Benchmark

### Test agent (interactive, turn-based)

```bash
export OPENAI_BASE_URL="http://your-endpoint/v1"
export OPENAI_API_KEY="your-api-key"
export GW_MODEL_NAME="your-model-name"

# 1 task — see full prompts and feedback
python scripts/run_benchmark.py --agent test --tier easy --max-tasks 1 \
    --max-turns 10 --max-calls-per-turn 8 --temperature 0.2 --verbose

# Full easy tier
python scripts/run_benchmark.py --agent test --tier easy \
    --max-turns 10 --max-calls-per-turn 8 --temperature 0.2 \
    --outfile results/test_easy.json

# All tiers
python scripts/run_benchmark.py --agent test --tier all \
    --max-turns 10 --max-calls-per-turn 8 --temperature 0.2 \
    --outfile results/test_full.json
```

### Running on a specific approximant dataset

Point `--data-dir` at the dataset you want. The evaluator reads the approximant from each task's `ground_truth.json` automatically:

```bash
python scripts/run_benchmark.py --agent test --tier easy \
    --data-dir data/seobnrv4 \
    --max-turns 10 --max-calls-per-turn 8 --temperature 0.2 \
    --outfile results/test_seobnrv4.json
```

---

## Evaluating an External Pipeline

Use `--agent external` to evaluate any external pipeline as a **black box**. The pipeline runs once per task, writes its parameter estimates to a JSON file, and the benchmark evaluates them. No turn loop, no feedback between submissions — single-shot evaluation.

### How it works

```
Benchmark writes input.json  →  external pipeline reads it, analyses data, writes output.json
                              →  benchmark reads output.json, submits to evaluator, records result
```

### Turn logic for external pipelines

The turn-based system (`--max-turns`, `--max-calls-per-turn`, feedback loop) **does not apply** to external pipelines. Each task gets exactly **one submission**. The pipeline has no access to criterion feedback — it runs completely independently and submits once. `--max-turns` is ignored.

This is intentional: external pipelines have their own internal loops, their own tools, their own analysis structure. The benchmark treats them as black boxes and only evaluates the final output.

### Step-by-step setup

**Step 1** — copy the template into your external pipeline repo:

```bash
cp external/pipeline_template.py /path/to/your/pipeline/run_agent.py
```

**Step 2** — open `run_agent.py` and replace the example analysis with your pipeline's actual code. The only requirements are:

- Read task data from the paths in `input.json`
- Write parameter estimates to `output_path` (the path is in `input.json`)

**Step 3** — run the benchmark:

```bash
python scripts/run_benchmark.py \
    --agent external \
    --pipeline-path /path/to/your/pipeline \
    --pipeline-entry run_agent.py \
    --tier easy --max-tasks 3 \
    --outfile results/external_easy.json
```

### What `input.json` contains

```json
{
    "task_id":          "synthetic_easy_001",
    "tier":             "easy",
    "difficulty_score": 3,
    "task_description": "...",
    "approximant":      "IMRPhenomD",
    "sample_rate_hz":   2048,
    "segment_duration_s": 16,
    "f_lower_hz":       20.0,
    "data_paths": {
        "strain_H1":  "/absolute/path/strain_H1.npy",
        "strain_L1":  "/absolute/path/strain_L1.npy",
        "psd_H1":     "/absolute/path/psd_H1.npy",
        "psd_L1":     "/absolute/path/psd_L1.npy",
        "psd_freqs":  "/absolute/path/psd_freqs.npy",
        "times":      "/absolute/path/times.npy"
    },
    "submission_format": { ... },
    "output_path": "/tmp/xxx/output.json"
}
```

### What `output.json` must contain

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

All 13 fields are required. Missing fields are filled with safe defaults. If the pipeline crashes or times out, a blank submission is filed automatically and the benchmark continues.

### External pipeline CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--pipeline-path` | required | Absolute path to the external pipeline repo root |
| `--pipeline-entry` | `run_agent.py` | Entry point script relative to `--pipeline-path` |
| `--pipeline-timeout` | `300` | Seconds to wait before killing the pipeline per task |

### Example with a downloaded pipeline

```bash
# Download a pipeline from GitHub
git clone https://github.com/someone/gw-agent-pipeline /tmp/gw-agent-pipeline

# Copy the entry point template
cp external/pipeline_template.py /tmp/gw-agent-pipeline/run_agent.py

# Edit run_agent.py to call the pipeline's actual code
# Then evaluate
python scripts/run_benchmark.py \
    --agent external \
    --pipeline-path /tmp/gw-agent-pipeline \
    --pipeline-entry run_agent.py \
    --tier easy \
    --outfile results/gw_agent_pipeline_easy.json
```

---

## CLI Arguments

### Task selection

| Argument | Default | Description |
|---|---|---|
| `--agent` | `test` | Agent to run: `test` or `external` |
| `--tier` | `all` | Which tier(s): `easy`, `medium`, `hard`, or `all` |
| `--max-tasks` | None (all) | Limit total tasks — useful for quick testing |
| `--data-dir` | `data/synthetic` | Path to dataset — change to use a different approximant dataset |
| `--outfile` | None | Save full results JSON to this path |
| `--verbose` | False | Print full prompts and feedback (interactive agents) or pipeline stdout (external) |

### Episode budget (interactive agents only — ignored for external)

| Argument | Default | Description |
|---|---|---|
| `--max-turns` | `10` | Submission attempts per episode. One turn = analysis code + one submission. |

### LLM parameters (test agent only)

| Argument | Default | Description |
|---|---|---|
| `--max-calls-per-turn` | `10` | Max `execute_python` calls per turn before blank submission forced |
| `--max-tokens` | `4096` | Max tokens per model response |
| `--temperature` | `0.3` | Model randomness: `0` = deterministic, `1` = creative |

### External pipeline parameters

| Argument | Default | Description |
|---|---|---|
| `--pipeline-path` | required | Path to external pipeline repo root |
| `--pipeline-entry` | `run_agent.py` | Entry point script relative to `--pipeline-path` |
| `--pipeline-timeout` | `300` | Seconds before pipeline is killed per task |

Run `python scripts/run_benchmark.py --help` to see all options with defaults.

---

## Understanding the Budget (Interactive Agents)

```
--max-turns 10          → 10 submission attempts = 10 turn_XX/ folders
--max-calls-per-turn 8  → up to 8 execute_python calls per turn

worst case total model calls = 10 × 8 = 80
```

For external pipelines: budget parameters are ignored. The pipeline runs once per task with no turn limit, no call limit, and no feedback loop. It has `--pipeline-timeout` seconds to complete its analysis and write `output.json`.

**Config printed at run start:**

```
# Interactive agent
============================================================
  GW Merger Bench
  Agent:           test
  max_turns:       10
  turn meaning:    analysis code (unlimited) + one submit = one turn
  max_calls_per_turn: 8
  temperature:     0.2
============================================================

# External pipeline
============================================================
  GW Merger Bench
  Agent:           external
  pipeline_path:   /tmp/gw-agent-pipeline
  pipeline_entry:  run_agent.py
  pipeline_timeout:300s
============================================================
```

**Recommended values for interactive agents:**

```bash
--max-turns 10 --max-calls-per-turn 8 --temperature 0.2
```

---

## Output Format

### Live output per task

```
[001/20] synthetic_easy_001   tier=easy   FAIL  crit=2/4  𝓜_err=78.33%  q_err=0.03  turns=1  t=12.4s
```

### Summary table

```
Tier      Pass          𝓜 err%    q err    Overlap    Stat✓Phys✗
--------  ------        --------  -------  ---------  ----------
easy      14/20 (70%)    8.32%    0.112      0.923        15%
medium     8/20 (40%)   18.74%    0.198      0.841        30%
hard       2/20 (10%)   45.21%    0.310      0.612        45%
overall   24/60 (40%)   24.09%    0.207      0.792        30%
```

| Column | Description |
|---|---|
| `Pass` | Tasks where all four criteria passed simultaneously |
| `𝓜 err%` | Mean chirp mass percentage error |
| `q err` | Mean mass ratio absolute error |
| `Overlap` | Mean noise-weighted waveform overlap (0–1) |
| `Stat✓Phys✗` | Tasks where waveform matched (≥ 0.90) but chirp mass failed — "good statistics ≠ good physics" gap |

### Results JSON structure

```json
{
  "task_id": "synthetic_easy_001",
  "tier": "easy",
  "passed": false,
  "metrics": {
    "ok_waveform_match": true,
    "ok_chirp_mass": false,
    "ok_mass_ratio": true,
    "ok_merger_type": true,
    "n_criteria_passed": 3,
    "waveform_overlap": 0.923,
    "waveform_overlap_threshold": 0.9,
    "chirp_mass_submitted": 32.4,
    "chirp_mass_true": 28.04,
    "chirp_mass_pct_err": 15.55,
    "mass_ratio_submitted": 0.74,
    "mass_ratio_true": 0.71,
    "mass_ratio_abs_err": 0.03,
    "merger_type_submitted": "BBH",
    "merger_type_true": "BBH",
    "stat_pass_phys_fail": true
  },
  "total_turns": 1,
  "submission_history": [ ... ]
}
```

---

## Adding a Custom Interactive Agent

Subclass `BaseAgent` in `scripts/run_benchmark.py` and register it alongside `test` and `external`:

```python
class MyAgent(BaseAgent):
    name = "my_agent"

    def reset(self, obs: dict):
        pass

    def act(self, env, obs: dict) -> dict:
        for turn in range(1, obs["max_turns"] + 1):
            result = env.execute_python("import numpy as np; ...")
            feedback = env.submit_action({
                "chirp_mass_Msun": 28.5,
                "mass1_Msun": 32.0, "mass2_Msun": 24.0,
                "mass_ratio": 0.75, "spin1z": 0.0, "spin2z": 0.0,
                "distance_Mpc": 450.0, "inclination_rad": 0.4,
                "ra_rad": 1.2, "dec_rad": -0.5,
                "network_snr": 20.0, "merger_type": "BBH", "confidence": 0.8,
            })
            if feedback["passed"] or feedback["episode_info"]["done"]:
                break
        return env.get_episode_summary()
```

Register and add to CLI choices in `scripts/run_benchmark.py`.

---

## Every Time You Return

```bash
cd ~/Desktop/code/BBH_interactive_datastet
source venv/bin/activate

export OPENAI_BASE_URL="http://your-endpoint/v1"
export OPENAI_API_KEY="your-api-key"
export GW_MODEL_NAME="your-model-name"
```
