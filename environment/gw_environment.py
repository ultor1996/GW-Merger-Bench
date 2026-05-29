"""
GW Merger Bench — Interactive Agentic Environment

Turn design:
  One TURN = unlimited execute_python calls + exactly one submit_action call.
  The turn ends when the agent submits. Feedback is returned.
  The agent then starts the next turn with that feedback in context.

  --max-turns controls how many submission attempts the agent gets.
  There is no separate max-submissions parameter — they are the same thing.

Episode ends when:
  - Agent passes all four criteria (conjunction gate)
  - Agent exhausts --max-turns
"""

import json
import os
import sys
import traceback
import io
import contextlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from evaluation.evaluator import GWEvaluator, EvaluationResult


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------
@dataclass
class EnvState:
    task_id: str
    tier: str
    difficulty_score: int
    task_dir: str
    turn: int = 0              # current turn number (increments on each submit)
    max_turns: int = 10        # max submission attempts
    done: bool = False
    best_result: Optional[EvaluationResult] = None
    submission_history: list = field(default_factory=list)
    # Code executed in the CURRENT turn (flushed to disk on submit)
    current_turn_code: List[str] = field(default_factory=list)
    current_turn_stdout: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PythonREPL — stateful execution sandbox
# ---------------------------------------------------------------------------
class PythonREPL:
    """
    Stateful Python execution environment.
    Persists variables across all calls within one episode.
    """

    def __init__(self, task_dir: str, output_dir: str = None):
        self._task_dir = task_dir
        self._output_dir = output_dir or task_dir
        self._globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
        }
        self._globals["TASK_DIR"]       = task_dir
        self._globals["OUTPUT_DIR"]     = self._output_dir
        self._globals["STRAIN_H1_PATH"] = os.path.join(task_dir, "strain_H1.npy")
        self._globals["STRAIN_L1_PATH"] = os.path.join(task_dir, "strain_L1.npy")
        self._globals["PSD_H1_PATH"]    = os.path.join(task_dir, "psd_H1.npy")
        self._globals["PSD_L1_PATH"]    = os.path.join(task_dir, "psd_L1.npy")
        self._globals["PSD_FREQS_PATH"] = os.path.join(task_dir, "psd_freqs.npy")
        self._globals["TIMES_PATH"]     = os.path.join(task_dir, "times.npy")

    def run(self, code: str, files_dir: str = None) -> Dict[str, Any]:
        """
        Execute code string. Returns stdout, stderr, success, error, saved_files.
        Any new files created in cwd are moved to files_dir if provided.
        """
        import glob, shutil

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cwd = os.getcwd()
        before = set(glob.glob(os.path.join(cwd, "*")))

        # Force Agg backend and ensure OUTPUT_DIR exists
        patched = (
            f"import os as _os; _os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
            f"import matplotlib as _mpl; _mpl.use('Agg')\n"
        ) + code

        try:
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stderr_buf):
                    exec(compile(patched, "<agent_code>", "exec"), self._globals)
            success, error = True, None
        except Exception:
            success, error = False, traceback.format_exc()

        # Detect and move new files
        after = set(glob.glob(os.path.join(cwd, "*")))
        new_files = [f for f in (after - before) if os.path.isfile(f)]
        saved = []
        if files_dir and new_files:
            os.makedirs(files_dir, exist_ok=True)
            for fpath in new_files:
                dest = os.path.join(files_dir, os.path.basename(fpath))
                try:
                    import shutil; shutil.move(fpath, dest)
                    saved.append(os.path.basename(fpath))
                except Exception:
                    pass

        return {
            "stdout":      stdout_buf.getvalue(),
            "stderr":      stderr_buf.getvalue(),
            "success":     success,
            "error":       error,
            "saved_files": saved,
        }


# ---------------------------------------------------------------------------
# Main Environment class
# ---------------------------------------------------------------------------
class GWEnvironment:
    """
    Interactive GW parameter estimation environment.

    Turn structure:
      - Agent calls execute_python() freely for analysis (no turn counter)
      - Agent calls submit_action() to end the turn and get feedback
      - submit_action() increments the turn counter
      - Episode ends when all criteria pass or max_turns reached
    """

    def __init__(
        self,
        task_dir: str,
        max_turns: int = 10,
        verbose: bool = False,
        run_dir: str = None,
    ):
        self.task_dir  = os.path.abspath(task_dir)
        self.max_turns = max_turns
        self.verbose   = verbose
        self.run_dir   = run_dir

        with open(os.path.join(self.task_dir, "task.json")) as f:
            self.task_meta = json.load(f)
        with open(os.path.join(self.task_dir, "ground_truth.json")) as f:
            self.ground_truth = json.load(f)

        self.evaluator = GWEvaluator(self.ground_truth, task_dir=self.task_dir)
        self._state: Optional[EnvState] = None
        self._repl:  Optional[PythonREPL] = None
        self._task_output_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        task_id = self.task_meta["task_id"]

        if self.run_dir:
            self._task_output_dir = os.path.join(self.run_dir, "tasks", task_id)
            os.makedirs(self._task_output_dir, exist_ok=True)
        else:
            self._task_output_dir = None

        self._state = EnvState(
            task_id=task_id,
            tier=self.task_meta["tier"],
            difficulty_score=self.task_meta["difficulty_score"],
            task_dir=self.task_dir,
            max_turns=self.max_turns,
        )
        self._repl = PythonREPL(
            self.task_dir,
            output_dir=self._task_output_dir or self.task_dir,
        )
        return self._build_initial_obs()

    # ------------------------------------------------------------------
    # Tool 1 — execute_python
    # Does NOT increment turn counter. Agent can call this freely.
    # ------------------------------------------------------------------
    def execute_python(self, code: str) -> Dict[str, Any]:
        if self._state is None or self._state.done:
            return {"error": "Episode not started or done. Call reset() first."}

        # Files for this turn go in a temp location until submit flushes them
        turn_files_dir = None
        if self._task_output_dir:
            # Staging folder for this turn's files
            turn_files_dir = os.path.join(
                self._task_output_dir, "_staging", "files"
            )
            os.makedirs(turn_files_dir, exist_ok=True)
            # Update OUTPUT_DIR in the sandbox so plt.savefig(OUTPUT_DIR/...)
            # saves directly into the correct turn folder — not the task root
            self._repl._globals["OUTPUT_DIR"] = turn_files_dir

        result = self._repl.run(code, files_dir=turn_files_dir)

        # Buffer code and output for this turn
        self._state.current_turn_code.append(code)
        self._state.current_turn_stdout.append(result.get("stdout", ""))

        if self.verbose:
            saved = result.get("saved_files", [])
            print(f"  [turn {self._state.turn+1} / analysis] "
                  f"execute_python success={result['success']}"
                  + (f" files={saved}" if saved else ""))

        return result

    # ------------------------------------------------------------------
    # Tool 2 — submit_action
    # Increments turn counter. Ends the current turn and returns feedback.
    # ------------------------------------------------------------------
    def submit_action(self, submission: Dict[str, Any]) -> Dict[str, Any]:
        if self._state is None or self._state.done:
            return {"error": "Episode not started or done. Call reset() first."}

        self._state.turn += 1
        turn_num = self._state.turn

        # Evaluate submission
        eval_result = self.evaluator.evaluate(submission)

        # Update best
        if (self._state.best_result is None or
                eval_result.n_criteria_passed > self._state.best_result.n_criteria_passed):
            self._state.best_result = eval_result

        # Flush turn to disk — rename staging → turn_XX
        if self._task_output_dir:
            turn_dir = os.path.join(
                self._task_output_dir, "turns", f"turn_{turn_num:02d}"
            )
            os.makedirs(turn_dir, exist_ok=True)

            # Save all code from this turn as one file
            full_code = "\n\n# --- next code block ---\n\n".join(
                self._state.current_turn_code
            )
            with open(os.path.join(turn_dir, "analysis_code.py"), "w") as f:
                f.write(full_code)

            # Save combined stdout
            full_stdout = "\n".join(self._state.current_turn_stdout)
            with open(os.path.join(turn_dir, "analysis_output.txt"), "w") as f:
                f.write(full_stdout)

            # Save the submission
            with open(os.path.join(turn_dir, "submission.json"), "w") as f:
                json.dump(submission, f, indent=2)

            # Save evaluator feedback
            with open(os.path.join(turn_dir, "feedback.json"), "w") as f:
                json.dump(eval_result.to_dict(), f, indent=2)

            # Move staged files into turn_dir/files/
            import shutil
            staging = os.path.join(self._task_output_dir, "_staging")
            if os.path.exists(staging):
                dest_files = os.path.join(turn_dir, "files")
                if os.path.exists(os.path.join(staging, "files")):
                    shutil.copytree(
                        os.path.join(staging, "files"), dest_files,
                        dirs_exist_ok=True
                    )
                shutil.rmtree(staging, ignore_errors=True)

        # Clear turn buffers for next turn
        self._state.current_turn_code = []
        self._state.current_turn_stdout = []

        # Record in submission history
        self._state.submission_history.append({
            "turn":       turn_num,
            "submission": submission,
            "result":     eval_result.to_dict(),
        })

        # Check termination
        if eval_result.passed:
            self._state.done = True
            status = "PASSED — all criteria satisfied."
        elif turn_num >= self._state.max_turns:
            self._state.done = True
            status = f"Turn limit reached ({self._state.max_turns}). Episode ended."
        else:
            remaining = self._state.max_turns - turn_num
            status = f"Not yet passing. {remaining} turn(s) remaining."

        if self.verbose:
            print(f"  [turn {turn_num}] submit_action "
                  f"passed={eval_result.passed} "
                  f"criteria={eval_result.n_criteria_passed}/4")

        return self._build_feedback(eval_result, status)

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------
    def get_episode_summary(self) -> Dict[str, Any]:
        if self._state is None:
            return {}

        br = self._state.best_result
        metrics = {}
        if br is not None:
            metrics = {
                "ok_waveform_match":   br.ok_waveform_match,
                "ok_chirp_mass":     br.ok_chirp_mass,
                "ok_mass_ratio":     br.ok_mass_ratio,
                "ok_merger_type":    br.ok_merger_type,
                "n_criteria_passed": br.n_criteria_passed,
                "chirp_mass_submitted":  round(br.chirp_mass_submitted, 3),
                "chirp_mass_true":       round(br.chirp_mass_true, 3),
                "chirp_mass_frac_err":   round(br.chirp_mass_frac_err, 4),
                "chirp_mass_pct_err":    round(br.chirp_mass_frac_err * 100, 2),
                "mass_ratio_submitted":  round(br.mass_ratio_submitted, 4),
                "mass_ratio_true":       round(br.mass_ratio_true, 4),
                "mass_ratio_abs_err":    round(br.mass_ratio_abs_err, 4),
                "waveform_overlap":      round(br.waveform_overlap, 4),
                "waveform_overlap_threshold": br.waveform_overlap_threshold,
                "merger_type_submitted": br.merger_type_submitted,
                "merger_type_true":      br.merger_type_true,
                "anchor_chirp_mass_from_freq_evo": round(br.anchor_chirp_mass_from_freq_evo, 3),
                "anchor_peak_freq_hz":             round(br.anchor_peak_freq_hz, 2),
                "stat_pass_phys_fail": (br.ok_waveform_match and not br.ok_chirp_mass),
            }

        summary = {
            "task_id":            self._state.task_id,
            "tier":               self._state.tier,
            "difficulty_score":   self._state.difficulty_score,
            "passed":             br.passed if br else False,
            "metrics":            metrics,
            "total_turns":        self._state.turn,
            "n_analysis_turns":   len(self._state.submission_history),
            "submission_history": self._state.submission_history,
        }

        if self._task_output_dir:
            with open(os.path.join(self._task_output_dir, "episode_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # Clean up any leftover staging folder — can remain if episode ended
            # without a submission (crash, interrupt, or max_calls_per_turn hit
            # before forced submission flushed it)
            import shutil
            staging = os.path.join(self._task_output_dir, "_staging")
            if os.path.exists(staging):
                # Move any staged files into a turn folder before deleting
                turns_dir = os.path.join(self._task_output_dir, "turns")
                existing = sorted(os.listdir(turns_dir)) if os.path.exists(turns_dir) else []
                next_turn = len(existing) + 1
                dest = os.path.join(turns_dir, f"turn_{next_turn:02d}", "files")
                staged_files = os.path.join(staging, "files")
                if os.path.exists(staged_files) and os.listdir(staged_files):
                    os.makedirs(dest, exist_ok=True)
                    shutil.copytree(staged_files, dest, dirs_exist_ok=True)
                shutil.rmtree(staging, ignore_errors=True)

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_initial_obs(self) -> Dict[str, Any]:
        return {
            "task_id":          self.task_meta["task_id"],
            "tier":             self.task_meta["tier"],
            "difficulty_score": self.task_meta["difficulty_score"],
            "task_description": self.task_meta["description"],
            "sample_rate_hz":   self.task_meta["sample_rate"],
            "segment_duration_s": self.task_meta["segment_duration"],
            "f_lower_hz":       self.task_meta["f_lower"],
            "detectors":        self.task_meta["detectors"],
            "approximant_hint": self.task_meta["approximant_hint"],
            "approximant": self.ground_truth.get("approximant", "IMRPhenomD"),
            "submission_format": self.task_meta["submission_format"],
            "data_paths": {
                "strain_H1":  os.path.join(self.task_dir, "strain_H1.npy"),
                "strain_L1":  os.path.join(self.task_dir, "strain_L1.npy"),
                "psd_H1":     os.path.join(self.task_dir, "psd_H1.npy"),
                "psd_L1":     os.path.join(self.task_dir, "psd_L1.npy"),
                "psd_freqs":  os.path.join(self.task_dir, "psd_freqs.npy"),
                "times":      os.path.join(self.task_dir, "times.npy"),
            },
            "tools_available": [
                {
                    "name": "execute_python",
                    "description": (
                        "Execute Python in a persistent sandbox. "
                        "Call this as many times as needed for analysis. "
                        "Does NOT count against your turn budget."
                    ),
                },
                {
                    "name": "submit_action",
                    "description": (
                        "Submit your parameter estimates. "
                        "This ENDS the current turn and returns criterion feedback. "
                        "You must call this exactly once per turn."
                    ),
                },
            ],
            "max_turns":    self.max_turns,
            "turn_meaning": "One turn = all your analysis code + one submission. "
                            f"You have {self.max_turns} turns total.",
        }

    def _build_feedback(self, result: EvaluationResult, status: str) -> Dict[str, Any]:
        # Per-criterion feedback:
        #   passed  → description only, no numbers
        #   failed  → description + error metric so agent knows how far off it was

        snr_entry = {
            "passed":      result.ok_waveform_match,
            "description": f"Passes if noise-weighted waveform overlap >= {result.waveform_overlap_threshold:.2f}",
        }
        if not result.ok_waveform_match:
            snr_entry["waveform_overlap"] = round(result.waveform_overlap, 4)
            snr_entry["overlap_needed"]   = result.waveform_overlap_threshold

        chirp_entry = {
            "passed":      result.ok_chirp_mass,
            "description": "Passes if chirp mass is within 5% of true value",
        }
        if not result.ok_chirp_mass:
            chirp_entry["chirp_mass_error_pct"] = round(result.chirp_mass_frac_err * 100, 2)

        ratio_entry = {
            "passed":      result.ok_mass_ratio,
            "description": "Passes if mass ratio is within 0.15 of true value",
        }
        if not result.ok_mass_ratio:
            ratio_entry["mass_ratio_abs_error"] = round(result.mass_ratio_abs_err, 4)

        merger_entry = {
            "passed":      result.ok_merger_type,
            "description": "Passes if merger type exactly matches (BBH / BNS / NSBH)",
        }
        if not result.ok_merger_type:
            merger_entry["submitted"] = result.merger_type_submitted
            merger_entry["expected"]  = "BBH, BNS, or NSBH"

        return {
            "status":  status,
            "passed":  result.passed,
            "turn":    self._state.turn,
            "turns_remaining": max(0, self._state.max_turns - self._state.turn),
            "criteria": {
                "ok_waveform_match": snr_entry,
                "ok_chirp_mass":   chirp_entry,
                "ok_mass_ratio":   ratio_entry,
                "ok_merger_type":  merger_entry,
            },
            "summary": {
                "n_criteria_passed": result.n_criteria_passed,
                "n_criteria_total":  4,
                "conjunction_gate":  result.passed,
            },
            "episode_info": {
                "turn":       self._state.turn,
                "done":       self._state.done,
            },
        }