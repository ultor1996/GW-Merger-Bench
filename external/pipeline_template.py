"""
GW Merger Bench — External Pipeline Entry Point Template

Copy this file into your pipeline repo as run_agent.py (or whatever
name you pass to --pipeline-entry) and implement the run() function.

The benchmark will call:
    python run_agent.py --input /tmp/xxx/input.json --output /tmp/xxx/output.json

Your script must:
  1. Read input.json (task description + absolute data paths)
  2. Analyse the data however you like
  3. Write output.json with parameter estimates

Nothing else is required. No imports from this benchmark needed.
"""

import argparse
import json
import numpy as np


def run(input_path: str, output_path: str):
    """
    Main entry point. Read input, analyse, write output.

    Parameters
    ----------
    input_path  : str — path to input.json written by the benchmark
    output_path : str — path where you must write output.json
    """

    # --- Read task input ---
    with open(input_path) as f:
        task = json.load(f)

    # What you receive:
    task_id      = task["task_id"]           # e.g. "synthetic_easy_001"
    tier         = task["tier"]              # "easy" / "medium" / "hard"
    description  = task["task_description"] # plain text description
    approximant  = task["approximant"]       # e.g. "IMRPhenomD"
    sample_rate  = task["sample_rate_hz"]    # 2048
    f_lower      = task["f_lower_hz"]        # 20.0

    # Absolute paths to numpy arrays — load directly with np.load
    strain_H1  = np.load(task["data_paths"]["strain_H1"])
    strain_L1  = np.load(task["data_paths"]["strain_L1"])
    psd_H1     = np.load(task["data_paths"]["psd_H1"])
    psd_L1     = np.load(task["data_paths"]["psd_L1"])
    psd_freqs  = np.load(task["data_paths"]["psd_freqs"])
    times      = np.load(task["data_paths"]["times"])

    print(f"Task: {task_id}  tier: {tier}  approximant: {approximant}")
    print(f"Strain shape: {strain_H1.shape}  PSD shape: {psd_H1.shape}")

    # ==================================================================
    # YOUR PIPELINE GOES HERE
    #
    # Analyse strain_H1, strain_L1, psd_H1, psd_L1, psd_freqs, times
    # and estimate the gravitational-wave parameters.
    #
    # You can use any libraries, any internal pipeline structure.
    # Just fill in the output dict below with your estimates.
    # ==================================================================

    # Example: rough chirp mass from ISCO frequency estimate
    # (replace with your actual analysis)
    n = len(strain_H1)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    fft_power = np.abs(np.fft.rfft(strain_H1)) ** 2
    band = (freqs > 30) & (freqs < 500)
    peak_freq = freqs[band][np.argmax(fft_power[band])]
    m_total_rough = 4400.0 / max(peak_freq, 1.0)
    chirp_mass_est = 0.87 * m_total_rough   # assumes q ~ 1

    print(f"Peak frequency: {peak_freq:.1f} Hz")
    print(f"Estimated chirp mass: {chirp_mass_est:.1f} M_sun")

    # ==================================================================
    # Write output.json — all fields required
    # ==================================================================
    output = {
        "chirp_mass_Msun": chirp_mass_est,
        "mass1_Msun":      m_total_rough * 0.55,
        "mass2_Msun":      m_total_rough * 0.45,
        "mass_ratio":      0.45 / 0.55,
        "spin1z":          0.0,
        "spin2z":          0.0,
        "distance_Mpc":    500.0,
        "inclination_rad": 0.4,
        "ra_rad":          1.5,
        "dec_rad":         -0.3,
        "network_snr":     15.0,
        "merger_type":     "BBH",
        "confidence":      0.5,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point — do not change this
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to input.json")
    parser.add_argument("--output", required=True, help="Path to write output.json")
    args = parser.parse_args()
    run(args.input, args.output)
