"""
GW Merger Bench — Qwen Agent
Uses an OpenAI-compatible endpoint (llama.cpp / vLLM / LM Studio style).

Usage:
  python scripts/run_benchmark.py --agent qwen --tier easy --max-tasks 3
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.system_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Lazy import so the file doesn't crash if openai isn't installed
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai package not installed. Run: pip install openai")


# ---------------------------------------------------------------------------
# Config — reads from environment variables only, no hardcoded fallbacks
# Set these before running:
#   export OPENAI_BASE_URL="http://your-endpoint/v1"
#   export OPENAI_API_KEY="your-api-key"
#   export GW_MODEL_NAME="your-model-name"   (optional)
# ---------------------------------------------------------------------------
BASE_URL  = os.environ.get("OPENAI_BASE_URL")
API_KEY   = os.environ.get("OPENAI_API_KEY")
MODEL     = os.environ.get("GW_MODEL_NAME")

MAX_TOKENS      = 4096
TEMPERATURE     = 0.3    # low — we want precise numerical estimates
MAX_REACT_TURNS    = 20   # max turns per episode
MAX_CALLS_PER_TURN = 10   # max model calls within one turn before forcing submit


# ---------------------------------------------------------------------------
# Qwen ReAct Agent
# ---------------------------------------------------------------------------
class TestAgent:
    """
    ReAct-style agent using the Qwen model at the custom endpoint.

    Loop:
      1. Send current conversation to model
      2. Parse model response for a tool call (execute_python or submit_action)
      3. Execute the tool, append result to conversation
      4. Repeat until episode done or max turns reached
    """

    name = "test_agent"

    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        max_react_turns: int = MAX_REACT_TURNS,
        max_calls_per_turn: int = MAX_CALLS_PER_TURN,
        verbose: bool = True,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("Install openai: pip install openai")

        if not base_url:
            raise RuntimeError(
                "OPENAI_BASE_URL is not set.\n"
                "Run: export OPENAI_BASE_URL=\"http://your-endpoint/v1\""
            )
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "Run: export OPENAI_API_KEY=\"your-api-key\""
            )
        if not model:
            raise RuntimeError(
                "GW_MODEL_NAME is not set.\n"
                "Run: export GW_MODEL_NAME=\"your-model-name\""
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_react_turns = max_react_turns
        self.max_calls_per_turn = max_calls_per_turn
        self.verbose = verbose
        self.messages = []

    def reset(self, obs: dict):
        """Called at episode start — build initial user message."""
        self.messages = []

        initial = self._build_initial_message(obs)
        self.messages.append({"role": "user", "content": initial})

        if self.verbose:
            sep = "=" * 70
            print(f"\n{sep}")
            print("SYSTEM PROMPT")
            print(sep)
            print(SYSTEM_PROMPT)
            print(f"\n{sep}")
            print("INITIAL USER MESSAGE")
            print(sep)
            print(initial)
            print(f"{sep}\n")

    def act(self, env, obs: dict) -> dict:
        """
        Run full episode using turn-based loop.

        Each turn:
          1. Model analyses data using execute_python() — as many calls as needed
          2. Model calls submit_action() once — ends the turn, returns feedback
          3. Feedback goes into context for next turn

        max_react_turns controls how many turns (= submissions) the agent gets.
        """
        max_turns = obs.get("max_turns", self.max_react_turns)

        for turn_num in range(1, max_turns + 1):
            if self.verbose:
                print(f"\n  === Turn {turn_num}/{max_turns} ===")

            # Inner loop: model calls execute_python freely, then must submit once
            submitted_this_turn = False
            inner_attempts = 0
            max_inner = self.max_calls_per_turn

            while not submitted_this_turn and inner_attempts < max_inner:
                inner_attempts += 1

                if self.verbose:
                    print(f"  [turn {turn_num} / call {inner_attempts}] calling model...")

                response_text = self._call_model()

                if self.verbose:
                    preview = response_text[:300].replace("\n", " ")
                    print(f"  [turn {turn_num} / call {inner_attempts}] {preview}...")

                self.messages.append({"role": "assistant", "content": response_text})

                tool_name, tool_args = self._parse_tool_call(response_text)

                if tool_name is None:
                    # No tool call — nudge model
                    nudge = (
                        "Please call execute_python() to continue your analysis, "
                        "or call submit_action() when ready to submit your estimates."
                    )
                    self.messages.append({"role": "user", "content": nudge})
                    continue

                if tool_name == "execute_python":
                    code = tool_args.get("code", "")
                    result = env.execute_python(code)
                    tool_output = self._format_python_result(result)

                    if self.verbose:
                        out_preview = tool_output[:200].replace("\n", " ")
                        print(f"  [turn {turn_num}] execute_python -> {out_preview}")

                    self.messages.append({"role": "user", "content": tool_output})

                elif tool_name == "submit_action":
                    submission = tool_args
                    feedback = env.submit_action(submission)
                    tool_output = self._format_feedback(feedback)

                    if self.verbose:
                        passed = feedback.get("passed", False)
                        n = feedback.get("summary", {}).get("n_criteria_passed", "?")
                        remaining = feedback.get("turns_remaining", "?")
                        print(f"  [turn {turn_num}] submitted -> "
                              f"passed={passed} criteria={n}/4 "
                              f"turns_remaining={remaining}")

                    self.messages.append({"role": "user", "content": tool_output})
                    submitted_this_turn = True

                    # Episode done (passed or turn limit)
                    if feedback.get("episode_info", {}).get("done", False):
                        return env.get_episode_summary()

                else:
                    self.messages.append({
                        "role": "user",
                        "content": f"Unknown tool '{tool_name}'. Use execute_python or submit_action."
                    })

            # If turn ended without a submission (inner loop exhausted), force submit
            if not submitted_this_turn:
                if self.verbose:
                    print(f"  [turn {turn_num}] forcing submission after {max_inner} calls")
                feedback = env.submit_action({
                    "chirp_mass_Msun": 0.0,
                    "mass1_Msun": 0.0,
                    "mass2_Msun": 0.0,
                    "mass_ratio": 0.5,
                    "spin1z": 0.0,
                    "spin2z": 0.0,
                    "distance_Mpc": 500.0,
                    "inclination_rad": 0.4,
                    "ra_rad": 1.5,
                    "dec_rad": -0.3,
                    "network_snr": 0.0,
                    "merger_type": "BBH",
                    "confidence": 0.0,
                })
                if feedback.get("episode_info", {}).get("done", False):
                    return env.get_episode_summary()

        return env.get_episode_summary()

    # ------------------------------------------------------------------
    # Private — LLM call
    # ------------------------------------------------------------------
    def _call_model(self) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            error_msg = f"[API ERROR: {e}]"
            print(f"  WARNING: model call failed: {e}")
            return error_msg

    # ------------------------------------------------------------------
    # Private — Tool call parser
    # ------------------------------------------------------------------
    def _parse_tool_call(self, text: str):
        """
        Parse a tool call from the model's text output.
        Supports two formats the model might use:

        Format 1 — JSON block:
          ```json
          {"tool": "execute_python", "code": "import numpy as np\n..."}
          ```

        Format 2 — function-call style:
          execute_python(code="import numpy as np\n...")
          submit_action({"chirp_mass_Msun": 28.5, ...})

        Format 3 — plain code block (assume execute_python):
          ```python
          import numpy as np
          ...
          ```
        """
        text_stripped = text.strip()

        # --- Format 1: explicit JSON tool call ---
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?"tool"\s*:.*?\})\s*```',
            text_stripped,
            re.DOTALL,
        )
        if json_match:
            try:
                obj = json.loads(json_match.group(1))
                tool = obj.pop("tool", None)
                if tool in ("execute_python", "submit_action"):
                    return tool, obj
            except json.JSONDecodeError:
                pass

        # --- Format 2a: submit_action with JSON dict ---
        submit_match = re.search(
            r'submit_action\s*\(\s*(\{.*?\})\s*\)',
            text_stripped,
            re.DOTALL,
        )
        if submit_match:
            try:
                args = json.loads(submit_match.group(1))
                return "submit_action", args
            except json.JSONDecodeError:
                pass

        # --- Format 2b: execute_python with code= kwarg ---
        exec_match = re.search(
            r'execute_python\s*\(\s*(?:code\s*=\s*)?["\'{](.*?)["\'}]\s*\)',
            text_stripped,
            re.DOTALL,
        )
        if exec_match:
            return "execute_python", {"code": exec_match.group(1)}

        # --- Format 3: raw python code block ---
        py_match = re.search(
            r'```python\s*(.*?)```',
            text_stripped,
            re.DOTALL,
        )
        if py_match:
            return "execute_python", {"code": py_match.group(1).strip()}

        # --- Format 4: inline submit dict (model says "I'll submit: {...}") ---
        inline_submit = re.search(
            r'(?:submit|submitting)[^\{]*(\{[^{}]*"chirp_mass_Msun"[^{}]*\})',
            text_stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if inline_submit:
            try:
                args = json.loads(inline_submit.group(1))
                return "submit_action", args
            except json.JSONDecodeError:
                pass

        return None, None

    # ------------------------------------------------------------------
    # Private — Message formatters
    # ------------------------------------------------------------------
    def _build_initial_message(self, obs: dict) -> str:
        return f"""You have a new gravitational-wave analysis task.

**Task ID**: {obs['task_id']}

**Task**: {obs['task_description']}

**Data files** (load with numpy):
  - Strain H1:   {obs['data_paths']['strain_H1']}
  - Strain L1:   {obs['data_paths']['strain_L1']}
  - PSD H1:      {obs['data_paths']['psd_H1']}
  - PSD L1:      {obs['data_paths']['psd_L1']}
  - PSD freqs:   {obs['data_paths']['psd_freqs']}
  - Time array:  {obs['data_paths']['times']}

Or use the pre-set variables: STRAIN_H1_PATH, STRAIN_L1_PATH, PSD_H1_PATH, PSD_L1_PATH, PSD_FREQS_PATH, TIMES_PATH

**Sample rate**: {obs['sample_rate_hz']} Hz  |  **Segment**: {obs['segment_duration_s']} s  |  **f_lower**: {obs['f_lower_hz']} Hz
**Detectors**: {obs['detectors']}

**Budget:**
  - You have **{obs['max_turns']} turns** total
  - Each turn = as many execute_python calls as you need + exactly one submit_action
  - Within each turn you can call execute_python up to **{self.max_calls_per_turn} times** before a submission is forced
  - A turn only ends when you call submit_action
  - Plan your analysis to fit within {self.max_calls_per_turn} code calls per turn, then submit

**Submission format** (pass as dict to submit_action):
{json.dumps(obs['submission_format'], indent=2)}

**To call a tool**, use one of these formats:

  execute_python:
  ```python
  import numpy as np
  strain = np.load(STRAIN_H1_PATH)
  print(strain.shape)
  ```

  submit_action:
  ```json
  {{"tool": "submit_action", "chirp_mass_Msun": 28.5, "mass1_Msun": 32.0, "mass2_Msun": 24.0, "mass_ratio": 0.75, "spin1z": 0.1, "spin2z": 0.05, "distance_Mpc": 450.0, "inclination_rad": 0.4, "ra_rad": 1.2, "dec_rad": -0.5, "network_snr": 18.5, "merger_type": "BBH", "confidence": 0.8}}
  ```

Start by loading and inspecting the strain data. Work step by step.
"""

    def _format_python_result(self, result: dict) -> str:
        parts = ["**execute_python result:**"]

        if result.get("stdout"):
            parts.append(f"```\n{result['stdout'].strip()}\n```")
        if result.get("stderr"):
            parts.append(f"⚠️ stderr:\n```\n{result['stderr'].strip()}\n```")
        if not result.get("success"):
            parts.append(f"❌ Error:\n```\n{result.get('error', 'unknown error').strip()}\n```")
        if result.get("env_message"):
            parts.append(f"⚠️ {result['env_message']}")

        if not result.get("stdout") and result.get("success"):
            parts.append("*(no output)*")

        return "\n".join(parts)

    def _format_feedback(self, feedback: dict) -> str:
        """
        Formats environment feedback into a string appended to the agent's
        conversation as a user message.

        Passed criteria  → description only, no numbers.
        Failed criteria  → description + error metric showing how far off.
        True values are never revealed.
        """
        parts = ["**submit_action feedback:**"]

        passed = feedback.get("passed", False)
        symbol = "✅" if passed else "❌"
        parts.append(f"{symbol} {feedback.get('status', '')}")

        parts.append("\nCriteria:")
        criteria = feedback.get("criteria", {})
        for name, info in criteria.items():
            tick = "✅" if info["passed"] else "❌"
            line = f"  {tick} {name}: {info['description']}"

            # On failure, show how far off so agent knows what to fix
            if not info["passed"]:
                if "waveform_overlap" in info:
                    line += f"  →  overlap: {info['waveform_overlap']:.3f} (need >= {info['overlap_needed']})"
                elif "chirp_mass_error_pct" in info:
                    line += f"  →  error: {info['chirp_mass_error_pct']}%"
                elif "mass_ratio_abs_error" in info:
                    line += f"  →  error: {info['mass_ratio_abs_error']} (absolute)"
                elif "submitted" in info:
                    line += f"  →  you submitted: {info['submitted']}"

            parts.append(line)

        summary   = feedback.get("summary", {})
        n         = summary.get("n_criteria_passed", 0)
        total     = summary.get("n_criteria_total", 4)
        turn      = feedback.get("turn", "?")
        remaining = feedback.get("turns_remaining", "?")
        ep        = feedback.get("episode_info", {})

        parts.append(f"\n{n}/{total} criteria passed.")
        parts.append(
            f"Turn: {turn}  |  "
            f"Turns remaining: {remaining}  |  "
            f"Done: {ep.get('done', False)}"
        )

        if not passed and not ep.get("done", False):
            parts.append(
                f"\nRefine your analysis and resubmit in the next turn. "
                f"You have up to {self.max_calls_per_turn} execute_python calls "
                f"before submission is forced."
            )

        text = "\n".join(parts)

        if self.verbose:
            sep = "=" * 70
            print(f"\n{sep}")
            print(f"FEEDBACK — Turn {turn}")
            print(sep)
            print(text)
            print(f"{sep}\n")

        return text