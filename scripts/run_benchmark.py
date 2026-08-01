"""
GW Merger Bench — Bare-bones benchmark runner.

Calls an external agent pipeline once per task, evaluates the final
submission against ground truth, saves a per-task JSON report.

No turn loop. No feedback to the agent. Single-shot evaluation only.
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


BLANK_SUBMISSION = {
    "chirp_mass_Msun":    0.0,
    "coalescence_time_s": 0.0,
}

REQUIRED_KEYS = {
    "chirp_mass_Msun", "coalescence_time_s",
}


def run_pipeline(pipeline_path, pipeline_entry, task_json,
                 task_dir, timeout, verbose) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.json")

        # times is NOT included — agent uses sample_rate from task.json
        pipeline_input = {
            "task_id":            task_json["task_id"],
            "task_description":   task_json["description"],
            "approximant":        task_json.get("approximant_hint", "IMRPhenomD"),
            "sample_rate_hz":     task_json["sample_rate"],
            "segment_duration_s": task_json["segment_duration"],
            "f_lower_hz":         task_json["f_lower"],
            "data_paths": {
                "strain_H1": os.path.abspath(os.path.join(task_dir, "strain_H1.npy")),
                "strain_L1": os.path.abspath(os.path.join(task_dir, "strain_L1.npy")),
                "psd_H1":    os.path.abspath(os.path.join(task_dir, "psd_H1.npy")),
                "psd_L1":    os.path.abspath(os.path.join(task_dir, "psd_L1.npy")),
                "psd_freqs": os.path.abspath(os.path.join(task_dir, "psd_freqs.npy")),
            },
            "submission_format": task_json.get("submission_format", {}),
            "output_path":       output_path,
        }

        with open(input_path, "w") as f:
            json.dump(pipeline_input, f, indent=2)

        if verbose:
            print(f"  [pipeline] input → {input_path}")

        entry        = os.path.join(pipeline_path, pipeline_entry)
        agent_python = os.path.join(pipeline_path, "venv", "bin", "python")
        if not os.path.exists(agent_python):
            agent_python = sys.executable
        cmd = [agent_python, entry, input_path]

        if verbose:
            print(f"  [pipeline] cmd: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd, cwd=pipeline_path,
                capture_output=not verbose, text=True, timeout=timeout,
            )
            if verbose and proc.stdout:
                print(proc.stdout[:2000])
            if proc.returncode != 0:
                print(f"  [pipeline] WARNING: exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  [pipeline] TIMEOUT after {timeout}s")
            return BLANK_SUBMISSION.copy()
        except Exception as e:
            print(f"  [pipeline] ERROR: {e}")
            return BLANK_SUBMISSION.copy()

        return _parse_output(output_path)


def _parse_output(output_path: str) -> dict:
    if not os.path.exists(output_path):
        print("  [pipeline] WARNING: output.json not found — blank submission")
        return BLANK_SUBMISSION.copy()
    try:
        with open(output_path) as f:
            output = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [pipeline] WARNING: invalid JSON: {e} — blank submission")
        return BLANK_SUBMISSION.copy()

    missing = REQUIRED_KEYS - set(output.keys())
    if missing:
        print(f"  [pipeline] WARNING: missing keys {missing} — using defaults")
        for key in missing:
            output[key] = BLANK_SUBMISSION[key]

    try:
        result = dict(output)  # keep everything, not just 2 keys
        result["chirp_mass_Msun"] = float(output["chirp_mass_Msun"])
        result["coalescence_time_s"] = float(output["coalescence_time_s"])
        return result
    except (ValueError, TypeError) as e:
        print(f"  [pipeline] WARNING: type error {e} — blank submission")
        return BLANK_SUBMISSION.copy()


def run_benchmark(args):
    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]
    tasks = load_tasks(args.data_dir, tiers, args.max_tasks, args.task_id, args.start_from_task_id)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_dir:
        run_dir = args.run_dir
    else:
        tier_str  = "all" if len(tiers) > 1 else tiers[0]
        run_dir   = os.path.join("results", f"{tier_str}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  GW Merger Bench")
    print(f"  Pipeline : {args.pipeline_entry}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Tier(s)  : {tiers}  |  Tasks: {len(tasks)}")
    print(f"  Run dir  : {run_dir}")
    print(f"{'='*60}\n")

    task_results = []

    for i, task_entry in enumerate(tasks, 1):
        task_id  = task_entry["task_id"]
        tier     = task_entry["tier"]
        task_dir = os.path.normpath(
            os.path.join(args.data_dir, "..", task_entry["path"])
        )

        with open(os.path.join(task_dir, "task.json")) as f:
            task_json = json.load(f)
        with open(os.path.join(task_dir, "ground_truth.json")) as f:
            ground_truth = json.load(f)

        t0 = time.time()

        submission = run_pipeline(
            pipeline_path=args.pipeline_path,
            pipeline_entry=args.pipeline_entry,
            task_json=task_json,
            task_dir=task_dir,
            timeout=args.pipeline_timeout,
            verbose=args.verbose,
        )

        evaluator = GWEvaluator(ground_truth, task_dir=task_dir)
        result    = evaluator.evaluate(submission)
        metrics   = result.to_dict()

        elapsed = round(time.time() - t0, 2)
        passed  = metrics["passed"]
        n_crit  = metrics["n_criteria_passed"]

        print(f"[{i:03d}/{len(tasks)}] {task_id:10s} tier={tier:6s} "
              f"{'PASS' if passed else 'FAIL'}  crit={n_crit}/{metrics['n_criteria_total']}   t={elapsed}s")

        task_result = {
            "task_id":    task_id,
            "tier":       tier,
            "passed":     passed,
            "elapsed_s":  elapsed,
            "submission": submission,
            "metrics":    metrics,
        }
        task_results.append(task_result)

        with open(os.path.join(run_dir, f"{task_id}.json"), "w") as f:
            json.dump(task_result, f, indent=2)

    # Merge in any pre-existing per-task JSONs already in run_dir (e.g. from
    # an earlier invocation before an interruption), so stats cover everything
    seen_task_ids = {r["task_id"] for r in task_results}
    for fname in os.listdir(run_dir):
        if fname.endswith(".json") and fname != "run_summary.json":
            tid = fname[:-5]
            if tid not in seen_task_ids:
                with open(os.path.join(run_dir, fname)) as f:
                    task_results.append(json.load(f))

    stats = _aggregate(task_results)
    _print_summary(stats)

    run_report = {
        "run_dir":      run_dir,
        "data_dir":     args.data_dir,
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

def load_tasks(data_dir: str, tiers: list, max_tasks: int = None, task_id: str = None,
                start_from_task_id: str = None) -> list:
    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index.json in {data_dir}. Run generate_dataset.py first."
        )
    with open(index_path) as f:
        index = json.load(f)

    tasks = [t for t in index["tasks"] if t["tier"] in tiers]

    if task_id:
        tasks = [t for t in tasks if t["task_id"] == task_id]
        if not tasks:
            raise ValueError(f"task_id '{task_id}' not found in tier(s) {tiers}")
        return tasks

    if start_from_task_id:
        tasks = [t for t in tasks if t["task_id"] >= start_from_task_id]
        if not tasks:
            raise ValueError(f"No tasks with task_id >= '{start_from_task_id}' in tier(s) {tiers}")

    if max_tasks:
        tasks = tasks[:max_tasks]
    return tasks

def _tier_stats(rs: list) -> dict:
    n        = len(rs)
    n_passed = sum(r["passed"] for r in rs)
    cms      = [r["metrics"].get("chirp_mass_frac_err", 1.0) for r in rs if r.get("metrics")]
    cts      = [r["metrics"].get("coalescence_time_abs_err", 1.0) for r in rs if r.get("metrics")]
    return {
        "n_tasks":                    n,
        "n_passed":                   n_passed,
        "pass_rate":                  round(n_passed / max(n, 1), 3),
        "mean_chirp_mass_pct_err":    round(sum(cms) / len(cms) * 100, 2) if cms else None,
        "mean_coalescence_time_err":  round(sum(cts) / len(cts), 4)       if cts else None,
    }

def _print_summary(stats: dict):
    print(f"\n{'Tier':<10} {'Pass':<14} {'Mc err%':>8}  {'t_c err(s)':>10}")
    print("-" * 46)
    for tier in ["easy", "medium", "hard", "overall"]:
        if tier not in stats:
            continue
        s   = stats[tier]
        ps  = f"{s['n_passed']}/{s['n_tasks']} ({s['pass_rate']*100:.0f}%)"
        cm  = f"{s['mean_chirp_mass_pct_err']}%" if s['mean_chirp_mass_pct_err'] is not None else "n/a"
        tc  = f"{s['mean_coalescence_time_err']}" if s['mean_coalescence_time_err'] is not None else "n/a"
        print(f"{tier:<10} {ps:<14} {cm:>8}  {tc:>10}")


def main():
    p = argparse.ArgumentParser(
        description="GW Merger Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pipeline-path",    required=True)
    p.add_argument("--pipeline-entry",   default="run.py")
    p.add_argument("--pipeline-timeout", type=int, default=300)
    p.add_argument("--tier",    default="all",
                   choices=["easy", "medium", "hard", "all"])
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--task-id",   default=None, help="Run only this specific task_id (e.g. '003')")
    p.add_argument("--start-from-task-id", default=None,
                   help="Skip tasks before this task_id; run it through the end of the selected tier(s)")
    p.add_argument("--run-dir", default=None,
                   help="Reuse this exact results folder instead of creating a new timestamped one -- "
                        "pass the same value across resumed invocations so everything lands together")
    p.add_argument("--data-dir",  default="data/IMRPhenomD")
    p.add_argument("--outfile",   default=None)
    p.add_argument("--verbose",   action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()