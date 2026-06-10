"""
GW Merger Bench — Bare-bones benchmark runner.

Calls an external agent pipeline once per task, evaluates the final
submission against ground truth, saves a per-task JSON report.

No turn loop. No feedback to the agent. Single-shot evaluation only.

Usage:
  python scripts/run_benchmark.py \
      --pipeline-path /path/to/your/pipeline \
      --pipeline-entry run_gw_benchmark.py \
      --tier easy \
      --outfile results/my_run.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import GWEvaluator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLANK_SUBMISSION = {
    "chirp_mass_Msun": 0.0,
    "mass1_Msun":      0.0,
    "mass2_Msun":      0.0,
    "mass_ratio":      0.5,
    "spin1z":          0.0,
    "spin2z":          0.0,
    "distance_Mpc":    500.0,
    "inclination_rad": 0.4,
    "ra_rad":          1.5,
    "dec_rad":         -0.3,
    "network_snr":     0.0,
    "merger_type":     "BBH",
    "confidence":      0.0,
}

REQUIRED_KEYS = {
    "chirp_mass_Msun", "mass1_Msun", "mass2_Msun", "mass_ratio",
    "spin1z", "spin2z", "distance_Mpc", "inclination_rad",
    "ra_rad", "dec_rad", "network_snr", "merger_type", "confidence",
}


# ---------------------------------------------------------------------------
# Pipeline runner — calls your agent as a subprocess
# ---------------------------------------------------------------------------
def run_pipeline(
    pipeline_path: str,
    pipeline_entry: str,
    task_json: dict,
    task_dir: str,
    timeout: int,
    verbose: bool,
) -> dict:
    """
    Write input.json, call the pipeline, read output.json.
    Returns the parsed submission dict (or BLANK_SUBMISSION on failure).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.json")

        # Build input.json — no tier, no difficulty_score
        pipeline_input = {
            "task_id":            task_json["task_id"],
            "task_description":   task_json["description"],
            "approximant":        task_json.get("approximant_hint", "IMRPhenomD"),
            "sample_rate_hz":     task_json["sample_rate"],
            "segment_duration_s": task_json["segment_duration"],
            "f_lower_hz":         task_json["f_lower"],
            "data_paths": {
                "strain_H1": os.path.join(task_dir, "strain_H1.npy"),
                "strain_L1": os.path.join(task_dir, "strain_L1.npy"),
                "psd_H1":    os.path.join(task_dir, "psd_H1.npy"),
                "psd_L1":    os.path.join(task_dir, "psd_L1.npy"),
                "psd_freqs": os.path.join(task_dir, "psd_freqs.npy"),
                "times":     os.path.join(task_dir, "times.npy"),
            },
            "submission_format":  task_json.get("submission_format", {}),
            "output_path":        output_path,
        }

        with open(input_path, "w") as f:
            json.dump(pipeline_input, f, indent=2)

        if verbose:
            print(f"  [pipeline] input.json → {input_path}")

        # Call pipeline
        entry = os.path.join(pipeline_path, pipeline_entry)
        cmd   = [sys.executable, entry, input_path]

        if verbose:
            print(f"  [pipeline] running: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                cwd=pipeline_path,
                capture_output=not verbose,
                text=True,
                timeout=timeout,
            )
            if verbose and proc.stdout:
                print(proc.stdout[:2000])
            if proc.returncode != 0:
                print(f"  [pipeline] WARNING: exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  [pipeline] TIMEOUT after {timeout}s")
            return BLANK_SUBMISSION.copy()
        except Exception as e:
            print(f"  [pipeline] ERROR: {e}")
            return BLANK_SUBMISSION.copy()

        # Read and validate output.json
        return _parse_output(output_path)


def _parse_output(output_path: str) -> dict:
    """Read output.json and validate. Falls back to BLANK_SUBMISSION."""
    if not os.path.exists(output_path):
        print(f"  [pipeline] WARNING: output.json not found — blank submission")
        return BLANK_SUBMISSION.copy()

    try:
        with open(output_path) as f:
            output = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [pipeline] WARNING: invalid JSON: {e} — blank submission")
        return BLANK_SUBMISSION.copy()

    # Fill missing keys
    missing = REQUIRED_KEYS - set(output.keys())
    if missing:
        print(f"  [pipeline] WARNING: missing keys {missing} — using defaults")
        for key in missing:
            output[key] = BLANK_SUBMISSION[key]

    # Type-safe conversion
    try:
        return {
            "chirp_mass_Msun": float(output["chirp_mass_Msun"]),
            "mass1_Msun":      float(output["mass1_Msun"]),
            "mass2_Msun":      float(output["mass2_Msun"]),
            "mass_ratio":      float(output["mass_ratio"]),
            "spin1z":          float(output["spin1z"]),
            "spin2z":          float(output["spin2z"]),
            "distance_Mpc":    float(output["distance_Mpc"]),
            "inclination_rad": float(output["inclination_rad"]),
            "ra_rad":          float(output["ra_rad"]),
            "dec_rad":         float(output["dec_rad"]),
            "network_snr":     float(output["network_snr"]),
            "merger_type":     str(output["merger_type"]).strip().upper(),
            "confidence":      float(output.get("confidence", 0.5)),
        }
    except (ValueError, TypeError) as e:
        print(f"  [pipeline] WARNING: type error {e} — blank submission")
        return BLANK_SUBMISSION.copy()


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------
def load_tasks(data_dir: str, tiers: list, max_tasks: int = None) -> list:
    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index.json in {data_dir}. Run generate_dataset.py first."
        )
    with open(index_path) as f:
        index = json.load(f)

    tasks = [t for t in index["tasks"] if t["tier"] in tiers]
    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(args):
    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]
    tasks = load_tasks(args.data_dir, tiers, args.max_tasks)

    # Timestamped results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tier_str  = "all" if len(tiers) > 1 else tiers[0]
    run_dir   = os.path.join("results", f"{tier_str}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GW Merger Bench")
    print(f"  Pipeline:  {args.pipeline_entry}")
    print(f"  Tier(s):   {tiers}  |  Tasks: {len(tasks)}")
    print(f"  Run dir:   {run_dir}")
    print(f"{'='*60}\n")

    task_results = []

    for i, task_entry in enumerate(tasks, 1):
        task_id  = task_entry["task_id"]
        tier     = task_entry["tier"]
        task_dir = os.path.join(args.data_dir, task_entry["path"])

        # Load task.json (public) and ground_truth.json (hidden from agent)
        with open(os.path.join(task_dir, "task.json")) as f:
            task_json = json.load(f)
        with open(os.path.join(task_dir, "ground_truth.json")) as f:
            ground_truth = json.load(f)

        t0 = time.time()

        # Run pipeline — agent never sees ground_truth
        submission = run_pipeline(
            pipeline_path=args.pipeline_path,
            pipeline_entry=args.pipeline_entry,
            task_json=task_json,
            task_dir=task_dir,
            timeout=args.pipeline_timeout,
            verbose=args.verbose,
        )

        # Evaluate submission against ground truth
        evaluator = GWEvaluator(ground_truth, task_dir=task_dir)
        result    = evaluator.evaluate(submission)
        metrics   = result.to_dict()

        elapsed = round(time.time() - t0, 2)
        passed  = metrics["passed"]
        n_crit  = metrics["n_criteria_passed"]

        print(f"[{i:03d}/{len(tasks)}] {task_id:35s} tier={tier:6s} "
              f"{'PASS' if passed else 'FAIL'}  "
              f"crit={n_crit}/4  "
              f"t={elapsed}s")

        task_result = {
            "task_id":    task_id,
            "tier":       tier,
            "passed":     passed,
            "elapsed_s":  elapsed,
            "submission": submission,
            "metrics":    metrics,
        }
        task_results.append(task_result)

        # Save per-task JSON immediately
        with open(os.path.join(run_dir, f"{task_id}.json"), "w") as f:
            json.dump(task_result, f, indent=2)

    # Aggregate + print summary
    stats = _aggregate(task_results)
    _print_summary(stats)

    # Save full run report
    run_report = {
        "run_dir":      run_dir,
        "pipeline":     args.pipeline_entry,
        "tiers":        tiers,
        "timestamp":    timestamp,
        "statistics":   stats,
        "task_results": task_results,
    }

    report_path = os.path.join(run_dir, "run_summary.json")
    with open(report_path, "w") as f:
        json.dump(run_report, f, indent=2)
    print(f"\nRun summary → {report_path}")

    if args.outfile:
        Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(run_report, f, indent=2)
        print(f"Also saved  → {args.outfile}")

    return run_report


# ---------------------------------------------------------------------------
# Aggregation + display
# ---------------------------------------------------------------------------
def _aggregate(task_results: list) -> dict:
    from collections import defaultdict
    by_tier = defaultdict(list)
    for r in task_results:
        by_tier[r["tier"]].append(r)

    stats = {}
    for tier in ["easy", "medium", "hard"]:
        rs = by_tier.get(tier, [])
        if rs:
            stats[tier] = _tier_stats(rs)

    stats["overall"] = _tier_stats(task_results)
    return stats


def _tier_stats(rs: list) -> dict:
    n        = len(rs)
    n_passed = sum(r["passed"] for r in rs)
    cms      = [r["metrics"].get("chirp_mass_frac_err", 1.0) for r in rs if r.get("metrics")]
    mrs      = [r["metrics"].get("mass_ratio_abs_err",  1.0) for r in rs if r.get("metrics")]
    overlaps = [r["metrics"].get("waveform_overlap",    0.0) for r in rs if r.get("metrics")]
    spf      = [r["metrics"].get("stat_pass_phys_fail", False) for r in rs if r.get("metrics")]
    return {
        "n_tasks":                  n,
        "n_passed":                 n_passed,
        "pass_rate":                round(n_passed / max(n, 1), 3),
        "mean_chirp_mass_pct_err":  round(sum(cms) / len(cms) * 100, 2) if cms else None,
        "mean_mass_ratio_abs_err":  round(sum(mrs) / len(mrs), 4)       if mrs else None,
        "mean_waveform_overlap":    round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        "stat_pass_phys_fail_rate": round(sum(spf) / len(spf), 3)       if spf else None,
    }


def _print_summary(stats: dict):
    print(f"\n{'Tier':<10} {'Pass':<14} {'Mc err%':>8}  {'q err':>6}  {'Overlap':>8}  {'Stat✓Phys✗':>10}")
    print("-" * 62)
    for tier in ["easy", "medium", "hard", "overall"]:
        if tier not in stats:
            continue
        s   = stats[tier]
        ps  = f"{s['n_passed']}/{s['n_tasks']} ({s['pass_rate']*100:.0f}%)"
        cm  = f"{s['mean_chirp_mass_pct_err']}%"  if s['mean_chirp_mass_pct_err']  is not None else "n/a"
        mr  = f"{s['mean_mass_ratio_abs_err']}"    if s['mean_mass_ratio_abs_err']  is not None else "n/a"
        ov  = f"{s['mean_waveform_overlap']}"       if s['mean_waveform_overlap']    is not None else "n/a"
        spf = f"{s['stat_pass_phys_fail_rate']*100:.0f}%" if s['stat_pass_phys_fail_rate'] is not None else "n/a"
        print(f"{tier:<10} {ps:<14} {cm:>8}  {mr:>6}  {ov:>8}  {spf:>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="GW Merger Bench — external pipeline evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pipeline-path",    required=True,
                   help="Absolute path to your agent pipeline repo root")
    p.add_argument("--pipeline-entry",   default="run_gw_benchmark.py",
                   help="Entry point script relative to --pipeline-path")
    p.add_argument("--pipeline-timeout", type=int, default=300,
                   help="Seconds before pipeline is killed per task")
    p.add_argument("--tier",    default="all",
                   choices=["easy", "medium", "hard", "all"])
    p.add_argument("--max-tasks", type=int, default=None,
                   help="Limit total tasks — useful for quick testing")
    p.add_argument("--data-dir",  default="data/synthetic",
                   help="Path to dataset directory")
    p.add_argument("--outfile",   default=None,
                   help="Also save full report to this path")
    p.add_argument("--verbose",   action="store_true",
                   help="Print pipeline stdout and full submission details")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()