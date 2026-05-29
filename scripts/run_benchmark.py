"""
GW Merger Bench — Benchmark Runner
Runs any agent on all (or a subset of) tasks and collects structured results.

Usage:
  python scripts/run_benchmark.py --agent test --tier easy
  python scripts/run_benchmark.py --agent external --pipeline-path /path/to/pipeline --tier easy
"""

import argparse
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.test_agent import TestAgent
    TEST_AVAILABLE = True
except Exception as _e:
    TEST_AVAILABLE = False

try:
    from agents.external_agent import ExternalPipelineAgent
    EXTERNAL_AVAILABLE = True
except Exception as _e:
    EXTERNAL_AVAILABLE = False

import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.gw_environment import GWEnvironment


# ---------------------------------------------------------------------------
# Agent interface — implement this for your LLM agent
# ---------------------------------------------------------------------------
class BaseAgent:
    """
    Base class for GW Merger Bench agents.
    Subclass this and implement act().
    """
    name = "base"

    def reset(self, obs: Dict[str, Any]):
        """Called at the start of each episode with the initial observation."""
        pass

    def act(
        self,
        env: GWEnvironment,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run one episode. Must call env.execute_python() and env.submit_action().
        Returns the episode summary from env.get_episode_summary().
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_benchmark(
    data_dir: str,
    agent: BaseAgent,
    tiers: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    max_turns: int = 10,
    verbose: bool = False,
    results_base_dir: str = "results",
    run_config: dict = None,
) -> Dict[str, Any]:

    if tiers is None:
        tiers = ["easy", "medium", "hard"]

    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index.json found in {data_dir}. Run generate_dataset.py first.")

    with open(index_path) as f:
        index = json.load(f)

    tasks = [t for t in index["tasks"] if t["tier"] in tiers]
    if max_tasks:
        tasks = tasks[:max_tasks]

    # Create timestamped run directory: results/<agent>_<tier>_<timestamp>/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tier_str = "all" if len(tiers) > 1 else tiers[0]
    run_name = f"{agent.name}_{tier_str}_{timestamp}"
    run_dir = os.path.join(results_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save run config
    if run_config:
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(run_config, f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Running agent '{agent.name}' on {len(tasks)} tasks "
          f"(tiers: {tiers})")

    results = []
    pass_by_tier = {"easy": [], "medium": [], "hard": []}

    for i, task_entry in enumerate(tasks):
        task_dir = os.path.join(data_dir, task_entry["path"])
        tier = task_entry["tier"]

        env = GWEnvironment(task_dir, max_turns=max_turns,
                            verbose=verbose, run_dir=run_dir)
        obs = env.reset()
        agent.reset(obs)

        t0 = time.time()
        try:
            summary = agent.act(env, obs)
        except Exception as e:
            import traceback
            summary = {
                "task_id": task_entry["task_id"],
                "tier": tier,
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        elapsed = time.time() - t0
        summary["elapsed_s"] = round(elapsed, 2)

        passed = summary.get("passed", False)
        pass_by_tier[tier].append(passed)
        results.append(summary)

        m = summary.get("metrics", {})
        status = "PASS" if passed else "FAIL"
        crit = m.get("n_criteria_passed", "?")
        cm_err = f"{m.get('chirp_mass_pct_err','?'):>5}%" if m else "     "
        mr_err = f"{m.get('mass_ratio_abs_err','?'):>5}" if m else "     "
        print(f"  [{i+1:03d}/{len(tasks)}] {task_entry['task_id']:30s} "
              f"tier={tier:6s} {status}  "
              f"crit={crit}/4  "
              f"𝓜_err={cm_err}  "
              f"q_err={mr_err}  "
              f"turns={summary.get('total_turns','?')}  "
              f"t={elapsed:.1f}s")

    # Aggregate statistics + per-metric averages
    stats = {}
    for tier in ["easy", "medium", "hard"]:
        results_tier = [r for r in results if r.get("tier") == tier]
        if not results_tier:
            continue
        passed_list = [r.get("passed", False) for r in results_tier]
        cms = [r["metrics"]["chirp_mass_pct_err"]  for r in results_tier if r.get("metrics")]
        mrs = [r["metrics"]["mass_ratio_abs_err"]  for r in results_tier if r.get("metrics")]
        snrs= [r["metrics"].get("waveform_overlap", 0.0) for r in results_tier if r.get("metrics")]
        spf = [r["metrics"]["stat_pass_phys_fail"]  for r in results_tier if r.get("metrics")]
        stats[tier] = {
            "n_tasks":                  len(results_tier),
            "n_passed":                 sum(passed_list),
            "pass_rate":                round(sum(passed_list) / len(passed_list), 3),
            "mean_chirp_mass_pct_err":  round(sum(cms)/len(cms), 2)  if cms  else None,
            "mean_mass_ratio_abs_err":  round(sum(mrs)/len(mrs), 4)  if mrs  else None,
            "mean_waveform_overlap":    round(sum(snrs)/len(snrs), 4) if snrs else None,
            "stat_pass_phys_fail_rate": round(sum(spf)/len(spf), 3)  if spf  else None,
        }

    all_results = results
    all_passed  = [r.get("passed", False) for r in all_results]
    all_cms  = [r["metrics"]["chirp_mass_pct_err"] for r in all_results if r.get("metrics")]
    all_mrs  = [r["metrics"]["mass_ratio_abs_err"] for r in all_results if r.get("metrics")]
    all_snrs = [r["metrics"].get("waveform_overlap", 0.0) for r in all_results if r.get("metrics")]
    all_spf  = [r["metrics"]["stat_pass_phys_fail"] for r in all_results if r.get("metrics")]
    stats["overall"] = {
        "n_tasks":                  len(all_passed),
        "n_passed":                 sum(all_passed),
        "pass_rate":                round(sum(all_passed) / max(len(all_passed), 1), 3),
        "mean_chirp_mass_pct_err":  round(sum(all_cms)/len(all_cms),   2) if all_cms  else None,
        "mean_mass_ratio_abs_err":  round(sum(all_mrs)/len(all_mrs),   4) if all_mrs  else None,
        "mean_waveform_overlap":    round(sum(all_snrs)/len(all_snrs), 2) if all_snrs else None,
        "stat_pass_phys_fail_rate": round(sum(all_spf)/len(all_spf),   3) if all_spf  else None,
    }

    print(f"\n--- Results for agent '{agent.name}' ---")
    print(f"  {'Tier':8s}  {'Pass':>6s}  {'𝓜 err%':>8s}  {'q err':>7s}  {'Overlap':>9s}  {'Stat✓Phys✗':>10s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}")
    for tier in ["easy", "medium", "hard", "overall"]:
        if tier not in stats:
            continue
        s = stats[tier]
        pass_str = f"{s['n_passed']}/{s['n_tasks']} ({s['pass_rate']*100:.0f}%)"
        cm   = f"{s['mean_chirp_mass_pct_err']}%" if s['mean_chirp_mass_pct_err'] is not None else "n/a"
        mr   = f"{s['mean_mass_ratio_abs_err']}"  if s['mean_mass_ratio_abs_err'] is not None else "n/a"
        snr  = f"{s['mean_waveform_overlap']}"     if s['mean_waveform_overlap'] is not None else "n/a"
        spf  = f"{s['stat_pass_phys_fail_rate']*100:.0f}%" if s['stat_pass_phys_fail_rate'] is not None else "n/a"
        print(f"  {tier:8s}  {pass_str:>6s}  {cm:>8s}  {mr:>7s}  {snr:>9s}  {spf:>10s}")

    final = {
        "agent": agent.name,
        "run_dir": run_dir,
        "tiers_evaluated": tiers,
        "statistics": stats,
        "task_results": results,
    }

    # Save run summary JSON
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nRun summary saved to {run_dir}/run_summary.json")

    return final


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run GW Merger Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Task selection ---
    parser.add_argument("--agent", type=str, default="test",
                        choices=["test", "external"],
                        help="Agent to run")
    parser.add_argument("--pipeline-path", type=str, default=None,
                        help="Path to external pipeline repo root (required when --agent external)")
    parser.add_argument("--pipeline-entry", type=str, default="run_agent.py",
                        help="Entry point script relative to --pipeline-path (default: run_agent.py)")
    parser.add_argument("--pipeline-timeout", type=int, default=300,
                        help="Seconds to wait for external pipeline before killing it (default: 300)")
    parser.add_argument("--data-dir", type=str, default="data/synthetic",
                        help="Path to synthetic dataset")
    parser.add_argument("--tier", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Which difficulty tier(s) to run")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit total number of tasks (useful for quick testing)")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Save full results JSON to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each model turn and tool result")

    # --- Environment / episode limits ---
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max turns per episode — each turn = one model call + one tool execution. Creates one turn_XX/ folder in results.")
    # --- LLM agent parameters ---
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per model response (test agent only)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Model temperature: 0=deterministic, 1=creative (test agent only)")
    parser.add_argument("--approximant", type=str, default="IMRPhenomD",
                        help="Waveform approximant the dataset was generated with (informational — stored in run_config)")
    parser.add_argument("--max-calls-per-turn", type=int, default=10,
                        help="Max model calls within one turn before a blank submission is forced (test agent only)")

    args = parser.parse_args()

    tiers = ["easy", "medium", "hard"] if args.tier == "all" else [args.tier]

    # Print active config so it's visible in logs
    print("=" * 60)
    print(f"  GW Merger Bench")
    print(f"  Agent:           {args.agent}")
    print(f"  Tier(s):         {tiers}")
    print(f"  max_turns:       {args.max_turns}")
    print(f"  turn meaning:    analysis code (unlimited) + one submit = one turn")
    if args.agent == "test":
        print(f"  max_tokens:         {args.max_tokens}")
        print(f"  temperature:        {args.temperature}")
        print(f"  max_calls_per_turn: {args.max_calls_per_turn}")
    elif args.agent == "external":
        print(f"  pipeline_path:   {args.pipeline_path}")
        print(f"  pipeline_entry:  {args.pipeline_entry}")
        print(f"  pipeline_timeout:{args.pipeline_timeout}s")
    if args.agent == "external":
        if not args.pipeline_path:
            print("ERROR: --pipeline-path is required when using --agent external")
            return
        agent = ExternalPipelineAgent(
            pipeline_path=args.pipeline_path,
            pipeline_entry=args.pipeline_entry,
            timeout=args.pipeline_timeout,
            verbose=args.verbose,
        )
    elif args.agent == "test":  # already handled above, this is a safety catch
        print(f"  max_tokens:         {args.max_tokens}")
        print(f"  temperature:        {args.temperature}")
        print(f"  max_calls_per_turn: {args.max_calls_per_turn}")
    print("=" * 60)

    # Build agent
    if args.agent == "external":
        if not args.pipeline_path:
            print("ERROR: --pipeline-path is required when using --agent external")
            return
        agent = ExternalPipelineAgent(
            pipeline_path=args.pipeline_path,
            pipeline_entry=args.pipeline_entry,
            timeout=args.pipeline_timeout,
            verbose=args.verbose,
        )
    elif args.agent == "test":
        if not TEST_AVAILABLE:
            print("ERROR: test agent unavailable. Run: pip install openai")
            return
        agent = TestAgent(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_react_turns=args.max_turns,
            max_calls_per_turn=args.max_calls_per_turn,
            verbose=args.verbose,
        )
    else:
        print(f"ERROR: unknown agent {args.agent}")
        return

    run_config = {
        "agent": args.agent,
        "tier": args.tier,
        "max_tasks": args.max_tasks,
        "max_turns": args.max_turns,

        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_calls_per_turn": args.max_calls_per_turn,
        "approximant": args.approximant,
        "data_dir": args.data_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    results = run_benchmark(
        data_dir=args.data_dir,
        agent=agent,
        tiers=tiers,
        max_tasks=args.max_tasks,
        max_turns=args.max_turns,
        verbose=args.verbose,
        results_base_dir="results",
        run_config=run_config,
    )

    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.outfile}")


if __name__ == "__main__":
    main()