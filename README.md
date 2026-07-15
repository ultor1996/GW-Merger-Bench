# GW Merger Bench

A benchmark for evaluating AI agents on gravitational-wave (GW) parameter estimation. Agents analyze synthetic binary black hole (BBH) strain data from LIGO-style detectors and recover physical parameters.

---

## Agent Submission Format

| Field | Description |
|---|---|
| `chirp_mass_Msun` | Chirp mass (M☉) — **scored** |
| `coalescence_time_s` | Merger time within the 16 s segment — **scored** |
| `mass1_Msun`, `mass2_Msun`, `mass_ratio` | Component masses / q = m2/m1 — reported, not currently scored |
| `network_snr` | Estimated SNR from matched filter — reported, not currently scored |
| `merger_type` | Exactly `"BBH"`, `"BNS"`, or `"NSBH"` — reported, not currently scored |

Spins, distance, sky location, and inclination are **not evaluated** — sky location (`ra`, `dec`, `polarisation`) is given directly to the agent in `task.json`'s `given_parameters` rather than estimated.

> **Note:** `mass_ratio` and `merger_type` are computed and reported by the pipeline, but `evaluation/evaluator.py`'s `GWEvaluator.evaluate()` currently only gates pass/fail on `chirp_mass_Msun` and `coalescence_time_s` (`n_criteria_total=2`). Helper methods for checking mass ratio, merger type, and waveform overlap exist in `evaluator.py` but are not called from `evaluate()`. If you want those to count toward scoring, they need to be wired in explicitly.

## What the Agent Receives

- `strain_H1.npy` / `strain_L1.npy` — 16 s strain time series
- `psd_H1.npy` / `psd_L1.npy` / `psd_freqs.npy` — detector noise PSDs
- `task.json` — sample rate, `f_lower`, approximant hint, segment duration, `given_parameters` (ra/dec/polarisation)

**Never given:** difficulty tier, true parameters, tolerances, `ground_truth.json`.

---

## Evaluation

Passes only if **both** hold:

| Criterion | Check | Tolerance |
|---|---|---|
| `ok_chirp_mass` | fractional error vs. `ground_truth.json`'s `chirp_mass` | `chirp_mass_tol_frac` (0.05 / 0.08 / 0.10 for easy/medium/hard) |
| `ok_coalescence_time` | absolute error vs. `ground_truth.json`'s `coalescence_time` | `coalescence_time_tol_s` (default 0.05 s if not set in ground truth) |

`n_criteria_total = 2`. Both `ok_chirp_mass` and `ok_coalescence_time` must be `True` for `passed=True`.

---

## Difficulty Tiers (current `DIFFICULTY_CONFIG`)

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Tasks per tier | 5 | 5 | 5 |
| `network_snr_range` | 20–35 | 12–20 | 8–12 |
| `total_mass_range` (M☉) | 40–73 | 25–73 | 10–73 |
| `mass_ratio_range` | 0.7–1.0 | 0.4–0.9 | 0.1–0.6 |
| `spin_magnitude_range` | 0.0 | 0.0 | 0.0 |
| `inclination_range` | 0.0 | 0.0 | 0.0 |
| `chirp_mass_tol_frac` | 0.05 | 0.08 | 0.10 |

Zero spin, face-on inclination for all tiers. Total mass is capped at 73 M☉ across every tier so the ISCO frequency stays comfortably above `f_lower`.

Default run = **15 tasks** (5/5/5). Change `n_tasks` per tier in `DIFFICULTY_CONFIG` to scale up.

---

## Repository Structure

```
GW_merger_bench/
├── scripts/
│   ├── generate_dataset.py
│   └── run_benchmark.py
├── evaluation/
│   └── evaluator.py
├── data/
│   └── <approximant>/
│       ├── index.json
│       └── <task_id>/
│           ├── strain_H1.npy, strain_L1.npy
│           ├── psd_H1.npy, psd_L1.npy, psd_freqs.npy
│           ├── task.json           (public)
│           └── ground_truth.json   (hidden)
└── results/
```

The agent-side pipeline (e.g. `physics_agent_harness/`, entry point `run_gw_multi.py`) is a separate repo, invoked as a subprocess by `run_benchmark.py`. It runs 4 stages per task: **Planning → Execution (full PE) → Critic → Reporting**, with up to `MAX_RETRIES=1` PE retries triggered by the critic.

## Generating Data

```bash
python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD --outdir data/IMRPhenomD_zerospin
```

Output lands at `data/IMRPhenomD_zerospin/IMRPhenomD/`. `--approximant` ∈ `{IMRPhenomD, SEOBNRv4, IMRPhenomXHM}`. Use the same `--seed` across approximants to keep physical parameters aligned.

**Injection placement:** the generator places each waveform's *numerical amplitude peak* (found via `argmax`, not the array's last sample) at the sampled `coalescence_time`. Earlier versions assumed a PyCBC time-domain waveform's last sample was the merger — this is false for `get_td_waveform` with IMRPhenomD (a frequency-domain-native model converted internally), which can return arrays several seconds longer than the physical inspiral, with the true peak sitting well before the array's end. If you see a fixed, suspiciously round offset (e.g. exactly ~5s) between recovered and true `coalescence_time` across many tasks, verify the peak-alignment placement is intact in `generate_one_event` before regenerating.

### `ground_truth.json` (hidden from agent) — actual current fields

```json
{
  "task_id": "000", "tier": "easy", "difficulty_score": 1,
  "mass1": 23.4696, "mass2": 19.3994, "chirp_mass": 18.5587, "mass_ratio": 0.8266,
  "spin1z": -0.0, "spin2z": -0.0,
  "distance": 1097.74, "inclination": 0.0, "ra": 1.4025, "dec": 0.7429, "polarisation": 2.1259,
  "coalescence_time": 13.1374, "network_snr": 29.591,
  "chirp_mass_from_freq_evo": 19.1138, "peak_frequency_hz": 102.59,
  "optimal_snr_H1": 24.101, "optimal_snr_L1": 17.17,
  "merger_type": "BBH", "approximant": "IMRPhenomD",
  "chirp_mass_tol_frac": 0.05, "mass_ratio_tol_abs": 0.15, "snr_tol_frac": 0.2
}
```

`chirp_mass_from_freq_evo` is an independent cross-check derived from measured df/dt near a reference frequency — used to validate waveform generation, not part of scoring. `mass_ratio_tol_abs` and `snr_tol_frac` are present but currently unused by `evaluate()` (see Evaluation section above).

---

## Running the Benchmark

```bash
python scripts/run_benchmark.py \
    --pipeline-path /path/to/pipeline \
    --pipeline-entry run_gw_multi.py \
    --data-dir data/IMRPhenomD_zerospin/IMRPhenomD \
    --tier all --pipeline-timeout 1800 \
    --outfile results/full_run.json --verbose
```

| Argument | Default | Description |
|---|---|---|
| `--pipeline-path` | required | Root of pipeline repo |
| `--pipeline-entry` | `run.py` | Entry point (e.g. `run_gw_multi.py`) |
| `--pipeline-timeout` | 300 | Seconds per task before hard kill (use 1800–4000 for full PE with retries) |
| `--tier` | `all` | `easy` / `medium` / `hard` / `all` |
| `--max-tasks` | none | Limit for quick tests (takes the first N tasks in tier order) |
| `--task-id` | none | Run only this specific `task_id` (e.g. `"003"`), bypassing `--max-tasks` |
| `--data-dir` | `data/IMRPhenomD` | Must contain `index.json` |
| `--outfile` | none | Save full run report |
| `--verbose` | off | Stream the pipeline's live output |

Missing output fields get safe defaults (`sanitise()` fills them in); crashes/timeouts/missing `output.json` record a blank submission and the run continues to the next task.

> **`--pipeline-timeout` vs. the pipeline's own internal budget:** the orchestrator (`run_gw_multi.py`) has its own `TOTAL_TIME_BUDGET_S` constant used to compute remaining-time prompts for the planning/critic agents. Keep this at or slightly below whatever `--pipeline-timeout` you pass here, or the orchestrator may greenlight a retry it doesn't actually have wall-clock time to finish before the harness kills the process.

### Summary output

```
Tier       Pass            Mc err%   t_c err(s)
----------------------------------------------
easy       1/1 (100%)        0.02%       0.012
overall    1/1 (100%)        0.02%       0.012
```

---

## Known Failure Modes (resolved during initial debugging)

| Symptom | Cause | Fix |
|---|---|---|
| `output.json not found — blank submission`, on every run | Pipeline never wrote `output.json` regardless of outcome | Write `sanitise(final_result)` to `output_path` unconditionally at the end of `main()`, wrapped in its own try/except |
| Critic can never `"accept"`, always `"retry"` regardless of fit quality | Dead `free_psi_used`/`free_ra_dec_used` gate in the critic prompt — condition can never be satisfied since no tool supports freeing sky location/psi | Removed; critic gate is now purely `chi2_reduced > 3.0 or log_bayes_factor < 0` |
| Retry barely changes anything (`walks`/`nact` unchanged across retries) | Orchestrator only applied `retry_nlive`/`retry_dlogz`, dropping `retry_sample`/`retry_walks`/`retry_nact` from the critic's response | Apply all five retry fields |
| `coalescence_time_s` missing from final output / `KeyError` | Not in `REQUIRED_KEYS`/`SAFE_DEFAULTS`, and not in the reporting agent's `final_answer` template | Added to both |
| `Invalid format specifier` crash in reporting agent | f-string dict literal used single `{ }` instead of `{{ }}` after an edit | Escape literal braces in the f-string template |
| `coalescence_time_s` off from true value by a fixed ~5s across all tasks | `generate_dataset.py` placed waveforms assuming the last sample of a PyCBC `get_td_waveform` array is the merger — false for this approximant, which returns arrays padded well past the physical peak | Peak-align placement via `np.argmax(np.abs(hp_arr))` instead of assuming `hp_arr[-1]` is merger |
| `check_waveform_residual`'s `chi2_reduced` pegged at 60–90 even for good fits | Chi² statistic used raw (un-normalized) overlaps that scale like SNR², inflating chi² by that factor | Normalize each bin's `z_i` by the template's own noise-weighted norm (`sigma`) before computing chi² |
| `chi2_reduced` still elevated (~4–6) even at exact true parameters | Sub-millisecond timing residual (from waveform-generation phase convention) causes multi-radian phase drift at high frequency — invisible in the time domain but highly visible to a phase-sensitive chi² test | Added a small local time-maximization (±10ms) inside `check_waveform_residual` before computing chi², instead of trusting the passed-in `merger_time_s` to sub-ms precision |
| Planning agent's Fisher-based uncertainty estimate always reports `σ≈0.0000` | `estimate_fisher_uncertainty`'s finite-difference/matrix-inversion step is broken — reports `valid=True` with degenerate near-zero sigma every time | Removed from the planning agent's tool list; sampler-config decisions now use only the matched-filter seed's SNR |
| `InterpreterError: 'classify_merger_type' is not among the explicitly allowed tools` | Reporting agent's prompt referenced `classify_merger_type` as if it were a registered tool, but it was only ever agent-defined inline | Registered as a real `@tool` in `gw_tools.py` |
| Hard to test a single specific task without re-running earlier ones | `run_benchmark.py` only supported `--tier`/`--max-tasks` (always starts from the first task in tier order) | Added `--task-id` to `load_tasks()`/`run_benchmark()` |