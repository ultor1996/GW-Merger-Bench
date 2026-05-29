"""
GW Merger Bench — External Pipeline Adapter
Evaluates any external agent pipeline as a black box.

The contract is simple:
  1. Benchmark writes  input.json  to a temp directory
  2. External pipeline is called:  python <entry_point> --input input.json --output output.json
  3. External pipeline writes output.json with parameter estimates
  4. Benchmark reads output.json and submits to evaluator

input.json structure (what the external pipeline receives):
{
    "task_id":          "synthetic_easy_001",
    "tier":             "easy",
    "difficulty_score": 3,
    "task_description": "...",
    "data_paths": {
        "strain_H1":  "/absolute/path/to/strain_H1.npy",
        "strain_L1":  "/absolute/path/to/strain_L1.npy",
        "psd_H1":     "/absolute/path/to/psd_H1.npy",
        "psd_L1":     "/absolute/path/to/psd_L1.npy",
        "psd_freqs":  "/absolute/path/to/psd_freqs.npy",
        "times":      "/absolute/path/to/times.npy"
    },
    "sample_rate_hz":     2048,
    "segment_duration_s": 16,
    "f_lower_hz":         20.0,
    "approximant":        "IMRPhenomD",
    "submission_format": { ... }
}

output.json structure (what the external pipeline must write):
{
    "chirp_mass_Msun": float,
    "mass1_Msun":      float,
    "mass2_Msun":      float,
    "mass_ratio":      float,
    "spin1z":          float,
    "spin2z":          float,
    "distance_Mpc":    float,
    "inclination_rad": float,
    "ra_rad":          float,
    "dec_rad":         float,
    "network_snr":     float,
    "merger_type":     "BBH" | "BNS" | "NSBH",
    "confidence":      float
}

Usage:
  python scripts/run_benchmark.py \
      --agent external \
      --pipeline-path /path/to/your/pipeline \
      --pipeline-entry run_agent.py \
      --tier easy --max-tasks 3
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Default submission used if the pipeline fails or produces invalid output
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
# ExternalPipelineAgent
# ---------------------------------------------------------------------------
class ExternalPipelineAgent:
    """
    Black-box adapter for external agent pipelines.

    Treats the pipeline as a single-shot process:
      - No turn loop
      - No feedback between submissions
      - Pipeline runs once per task, returns one set of estimates
      - Benchmark submits those estimates and evaluates

    Args:
        pipeline_path: path to the pipeline repo root
        pipeline_entry: entry point script relative to pipeline_path
                        defaults to "run_agent.py"
        timeout:        seconds to wait before killing the pipeline
                        defaults to 300 (5 minutes)
        python_bin:     Python interpreter to use (defaults to current venv)
        verbose:        print pipeline stdout/stderr
    """

    name = "external_pipeline"

    def __init__(
        self,
        pipeline_path: str,
        pipeline_entry: str = "run_agent.py",
        timeout: int = 300,
        python_bin: str = None,
        verbose: bool = False,
    ):
        self.pipeline_path  = os.path.abspath(pipeline_path)
        self.pipeline_entry = pipeline_entry
        self.timeout        = timeout
        self.python_bin     = python_bin or sys.executable
        self.verbose        = verbose

        entry = os.path.join(self.pipeline_path, self.pipeline_entry)
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline path not found: {self.pipeline_path}")
        if not os.path.exists(entry):
            raise FileNotFoundError(
                f"Entry point not found: {entry}\n"
                f"Set --pipeline-entry to the correct script name."
            )

    def reset(self, obs: dict):
        pass  # stateless — no conversation history to clear

    def act(self, env, obs: dict) -> dict:
        """
        Run one episode:
          1. Write input.json for the pipeline
          2. Call the pipeline as a subprocess
          3. Read output.json
          4. Submit to evaluator
          5. Return episode summary
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            input_path  = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")

            # --- Write input.json ---
            pipeline_input = {
                "task_id":            obs["task_id"],
                "tier":               obs["tier"],
                "difficulty_score":   obs["difficulty_score"],
                "task_description":   obs["task_description"],
                "data_paths":         obs["data_paths"],
                "sample_rate_hz":     obs["sample_rate_hz"],
                "segment_duration_s": obs["segment_duration_s"],
                "f_lower_hz":         obs["f_lower_hz"],
                "approximant":        obs.get("approximant", "IMRPhenomD"),
                "submission_format":  obs["submission_format"],
                "output_path":        output_path,
            }

            with open(input_path, "w") as f:
                json.dump(pipeline_input, f, indent=2)

            if self.verbose:
                print(f"\n  [external] input.json written to {input_path}")

            # --- Call the pipeline ---
            entry = os.path.join(self.pipeline_path, self.pipeline_entry)
            cmd = [
                self.python_bin, entry,
                "--input",  input_path,
                "--output", output_path,
            ]

            if self.verbose:
                print(f"  [external] running: {' '.join(cmd)}")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.pipeline_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                if self.verbose:
                    if proc.stdout:
                        print(f"  [external] stdout:\n{proc.stdout[:1000]}")
                    if proc.stderr:
                        print(f"  [external] stderr:\n{proc.stderr[:500]}")

                if proc.returncode != 0:
                    print(f"  [external] WARNING: pipeline exited with code {proc.returncode}")

            except subprocess.TimeoutExpired:
                print(f"  [external] WARNING: pipeline timed out after {self.timeout}s — submitting blank")
                env.submit_action(BLANK_SUBMISSION)
                return env.get_episode_summary()

            except Exception as e:
                print(f"  [external] WARNING: pipeline call failed: {e}")
                env.submit_action(BLANK_SUBMISSION)
                return env.get_episode_summary()

            # --- Read output.json ---
            submission = self._parse_output(output_path)

            if self.verbose:
                print(f"  [external] submission: {json.dumps(submission, indent=2)}")

            # --- Submit once ---
            feedback = env.submit_action(submission)

            if self.verbose:
                passed = feedback.get("passed", False)
                n      = feedback.get("summary", {}).get("n_criteria_passed", "?")
                print(f"  [external] result: passed={passed} criteria={n}/4")

        return env.get_episode_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _parse_output(self, output_path: str) -> dict:
        """
        Read and validate output.json from the pipeline.
        Falls back to BLANK_SUBMISSION if file is missing or malformed.
        """
        if not os.path.exists(output_path):
            print(f"  [external] WARNING: output.json not found at {output_path} — submitting blank")
            return BLANK_SUBMISSION.copy()

        try:
            with open(output_path) as f:
                output = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [external] WARNING: output.json is invalid JSON: {e} — submitting blank")
            return BLANK_SUBMISSION.copy()

        # Check required keys
        missing = REQUIRED_KEYS - set(output.keys())
        if missing:
            print(f"  [external] WARNING: output.json missing keys: {missing}")
            print(f"  [external] Filling missing keys with defaults")
            for key in missing:
                output[key] = BLANK_SUBMISSION[key]

        # Type-safe conversion
        try:
            submission = {
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
            print(f"  [external] WARNING: type conversion failed: {e} — submitting blank")
            return BLANK_SUBMISSION.copy()

        return submission
