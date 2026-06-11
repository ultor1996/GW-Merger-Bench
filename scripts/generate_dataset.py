"""
GW Merger Bench — Dataset Generator
Generates 300 synthetic BBH injection tasks (100 easy / 100 medium / 100 hard).

Each task is saved as:
  data/{approximant}/{task_id}/
      strain_H1.npy       — detector strain time series (H1)
      strain_L1.npy       — detector strain time series (L1)
      psd_H1.npy          — noise PSD used (H1)
      psd_L1.npy          — noise PSD used (L1)
      psd_freqs.npy       — PSD frequency axis
      times.npy           — GPS-relative time array
      task.json           — task metadata (public, no true params, no tier)
      ground_truth.json   — true parameters + tier (hidden from agent)

All tasks live in one flat folder under the approximant subfolder.
Tier is stored only in ground_truth.json — never given to the agent.

Usage:
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD
  python scripts/generate_dataset.py --seed 42 --approximant SEOBNRv4
"""

import argparse
import json
import os
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple

# ---------------------------------------------------------------------------
# PyCBC imports
# ---------------------------------------------------------------------------
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries, FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    print("WARNING: pycbc not found. Install with: pip install pycbc")


# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------
DIFFICULTY_CONFIG = {
    "easy": {
        "n_tasks":               5,
        "network_snr_range":     (20.0, 35.0),
        "total_mass_range":      (40.0, 80.0),
        "mass_ratio_range":      (0.7, 1.0),
        "spin_magnitude_range":  (0.0, 0.1),
        "inclination_range":     (0.0, 0.3),
        "difficulty_score_range":(1, 3),
    },
    "medium": {
        "n_tasks":               5,
        "network_snr_range":     (12.0, 20.0),
        "total_mass_range":      (25.0, 120.0),
        "mass_ratio_range":      (0.4, 0.9),
        "spin_magnitude_range":  (0.0, 0.5),
        "inclination_range":     (0.0, 1.0),
        "difficulty_score_range":(4, 7),
    },
    "hard": {
        "n_tasks":               5,
        "network_snr_range":     (8.0, 12.0),
        "total_mass_range":      (10.0, 200.0),
        "mass_ratio_range":      (0.1, 0.6),
        "spin_magnitude_range":  (0.3, 0.9),
        "inclination_range":     (0.5, np.pi / 2),
        "difficulty_score_range":(8, 10),
    },
}

SAMPLE_RATE      = 2048
SEGMENT_DURATION = 16
F_LOWER          = 20.0
APPROXIMANT      = "IMRPhenomD"   # overridden by CLI


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrueParams:
    """Ground truth — never shown to the agent."""
    task_id:                  str
    tier:                     str
    difficulty_score:         int
    mass1:                    float
    mass2:                    float
    chirp_mass:               float
    mass_ratio:               float
    spin1z:                   float
    spin2z:                   float
    distance:                 float
    inclination:              float
    ra:                       float
    dec:                      float
    polarisation:             float
    coalescence_time:         float
    network_snr:              float
    chirp_mass_from_freq_evo: float
    peak_frequency_hz:        float
    optimal_snr_H1:           float
    optimal_snr_L1:           float
    merger_type:              str
    approximant:              str
    chirp_mass_tol_frac:      float
    mass_ratio_tol_abs:       float
    snr_tol_frac:             float


@dataclass
class TaskMetadata:
    """Public task description given to the agent. No tier or difficulty."""
    task_id:           str
    description:       str
    sample_rate:       int
    segment_duration:  float
    f_lower:           float
    detectors:         list
    approximant_hint:  str
    submission_format: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chirp_mass(m1: float, m2: float) -> float:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def _chirp_mass_from_dfdt(f: float, dfdt: float) -> float:
    G_over_c3 = 4.925491025543576e-06
    Mchirp_s = (5.0 / 96.0 * np.pi ** (-8.0 / 3.0) *
                f ** (-11.0 / 3.0) * dfdt) ** (3.0 / 5.0)
    return Mchirp_s / G_over_c3


def _colored_noise(psd_vals, psd_freqs, n_samples, sample_rate, seed):
    """Colored Gaussian noise from PSD — numpy only, no lal dependency."""
    rng_n      = np.random.default_rng(seed)
    flen       = n_samples // 2 + 1
    freqs      = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    psd_interp = np.interp(freqs, psd_freqs, psd_vals, left=1e-40, right=1e-40)
    psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
    sigma_f    = np.sqrt(psd_interp * sample_rate / 2)
    noise_f    = (rng_n.standard_normal(flen) +
                  1j * rng_n.standard_normal(flen)) * sigma_f
    noise_f[0]  = noise_f[0].real
    noise_f[-1] = noise_f[-1].real
    return np.fft.irfft(noise_f, n=n_samples).astype(np.float64)


# ---------------------------------------------------------------------------
# Generate one event
# ---------------------------------------------------------------------------
def generate_one_event(
    task_id: str,
    tier: str,
    cfg: dict,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[TrueParams, TaskMetadata, dict]:

    if not PYCBC_AVAILABLE:
        raise RuntimeError("pycbc is required.")

    total_mass  = rng.uniform(*cfg["total_mass_range"])
    q           = rng.uniform(*cfg["mass_ratio_range"])
    m1          = total_mass / (1.0 + q)
    m2          = total_mass * q / (1.0 + q)
    if m1 < m2:
        m1, m2 = m2, m1

    spin1z      = rng.uniform(*cfg["spin_magnitude_range"]) * rng.choice([-1, 1])
    spin2z      = rng.uniform(*cfg["spin_magnitude_range"]) * rng.choice([-1, 1])
    inclination = rng.uniform(*cfg["inclination_range"])
    ra          = rng.uniform(0, 2 * np.pi)
    dec         = rng.uniform(-np.pi / 2, np.pi / 2)
    polarisation= rng.uniform(0, np.pi)
    coa_phase   = rng.uniform(0, 2 * np.pi)
    coa_time_offset = SEGMENT_DURATION * 0.67

    dt    = 1.0 / SAMPLE_RATE
    flen  = int(SEGMENT_DURATION * SAMPLE_RATE / 2) + 1

    # Waveform
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=m1, mass2=m2,
        spin1z=spin1z, spin2z=spin2z,
        inclination=inclination, coa_phase=coa_phase,
        delta_t=dt, f_lower=F_LOWER, distance=100.0,
    )

    # PSD
    delta_f = 1.0 / SEGMENT_DURATION
    psd_H1  = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_L1  = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    # Detector projection
    det_H1  = Detector("H1")
    det_L1  = Detector("L1")
    ref_gps = 1264316116.0
    gps_coa = ref_gps + coa_time_offset
    fp_H1, fc_H1 = det_H1.antenna_pattern(ra, dec, polarisation, gps_coa)
    fp_L1, fc_L1 = det_L1.antenna_pattern(ra, dec, polarisation, gps_coa)

    n_samples = int(SEGMENT_DURATION * SAMPLE_RATE)

    def project_and_resize(hp, hc, fp, fc, n):
        sig      = fp * hp + fc * hc
        sig_arr  = np.zeros(n)
        coa_idx  = int(coa_time_offset * SAMPLE_RATE)
        end_idx  = min(coa_idx, len(sig))
        start_src= len(sig) - end_idx
        sig_arr[coa_idx - end_idx: coa_idx] = np.array(sig)[start_src:]
        return sig_arr

    sig_H1 = project_and_resize(hp, hc, fp_H1, fc_H1, n_samples)
    sig_L1 = project_and_resize(hp, hc, fp_L1, fc_L1, n_samples)

    # Optimal SNR at 100 Mpc
    try:
        opt_snr_H1_100 = float(sigma(TimeSeries(sig_H1, delta_t=dt),
                                     psd=psd_H1, low_frequency_cutoff=F_LOWER))
        opt_snr_L1_100 = float(sigma(TimeSeries(sig_L1, delta_t=dt),
                                     psd=psd_L1, low_frequency_cutoff=F_LOWER))
    except Exception:
        opt_snr_H1_100 = opt_snr_L1_100 = 10.0

    network_snr_100 = np.sqrt(opt_snr_H1_100 ** 2 + opt_snr_L1_100 ** 2)
    target_snr      = rng.uniform(*cfg["network_snr_range"])
    distance        = 100.0 * network_snr_100 / target_snr if network_snr_100 > 0 else 500.0
    scale           = 100.0 / distance
    sig_H1         *= scale
    sig_L1         *= scale
    opt_snr_H1      = opt_snr_H1_100 * scale
    opt_snr_L1      = opt_snr_L1_100 * scale
    network_snr     = np.sqrt(opt_snr_H1 ** 2 + opt_snr_L1 ** 2)

    # PSD arrays (must be built before noise generation)
    psd_freqs_arr = np.linspace(0, SAMPLE_RATE / 2, flen)
    psd_vals_H1   = np.array(psd_H1)
    psd_vals_L1   = np.array(psd_L1)

    # Colored Gaussian noise
    noise_arr_H1 = _colored_noise(psd_vals_H1, psd_freqs_arr, n_samples, SAMPLE_RATE,
                                  seed=abs(hash(task_id + "H1")) % (2 ** 31))
    noise_arr_L1 = _colored_noise(psd_vals_L1, psd_freqs_arr, n_samples, SAMPLE_RATE,
                                  seed=abs(hash(task_id + "L1")) % (2 ** 31))

    strain_H1 = noise_arr_H1 + sig_H1
    strain_L1 = noise_arr_L1 + sig_L1

    # Anchor values
    Mc        = _chirp_mass(m1, m2)
    isco_freq = 4400.0 / (m1 + m2)
    G_over_c3 = 4.925491025543576e-06
    Mc_s      = Mc * G_over_c3
    f_ref     = 100.0
    dfdt_ref  = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * Mc_s ** (5.0 / 3.0) * f_ref ** (11.0 / 3.0)
    Mc_anchor = _chirp_mass_from_dfdt(f_ref, dfdt_ref)

    lo, hi    = cfg["difficulty_score_range"]
    diff_score= rng.randint(lo, hi)

    true_params = TrueParams(
        task_id=task_id, tier=tier, difficulty_score=diff_score,
        mass1=round(m1, 4), mass2=round(m2, 4),
        chirp_mass=round(Mc, 4), mass_ratio=round(q, 4),
        spin1z=round(spin1z, 4), spin2z=round(spin2z, 4),
        distance=round(distance, 2), inclination=round(inclination, 4),
        ra=round(ra, 4), dec=round(dec, 4), polarisation=round(polarisation, 4),
        coalescence_time=round(coa_time_offset, 4),
        network_snr=round(network_snr, 3),
        chirp_mass_from_freq_evo=round(Mc_anchor, 4),
        peak_frequency_hz=round(isco_freq, 2),
        optimal_snr_H1=round(opt_snr_H1, 3),
        optimal_snr_L1=round(opt_snr_L1, 3),
        merger_type="BBH", approximant=APPROXIMANT,
        chirp_mass_tol_frac=0.05, mass_ratio_tol_abs=0.15, snr_tol_frac=0.20,
    )

    task_meta = TaskMetadata(
        task_id=task_id,
        description=(
            "A gravitational-wave strain signal has been recorded by the H1 and L1 detectors. "
            f"The segment is {SEGMENT_DURATION}s long at {SAMPLE_RATE} Hz. "
            "Your task is to detect the signal, estimate the chirp mass, component masses, "
            "mass ratio, spins, distance, and sky location, and classify the merger type. "
            "Submit your parameter estimates."
        ),
        sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, f_lower=F_LOWER,
        detectors=["H1", "L1"], approximant_hint=APPROXIMANT,
        submission_format={
            "chirp_mass_Msun": "float",
            "mass1_Msun":      "float",
            "mass2_Msun":      "float",
            "mass_ratio":      "float — m2/m1, in (0,1]",
            "network_snr":     "float — your estimated SNR",
            "merger_type":     "str — one of BBH / BNS / NSBH",
        },
    )

    arrays = {
        "strain_H1": strain_H1, "strain_L1": strain_L1,
        "psd_H1":    psd_vals_H1, "psd_L1":  psd_vals_L1,
        "psd_freqs": psd_freqs_arr, "times":  np.linspace(0, SEGMENT_DURATION, n_samples, endpoint=False),
    }

    return true_params, task_meta, arrays


# ---------------------------------------------------------------------------
# Save one task
# ---------------------------------------------------------------------------
def save_task(base_outdir: str, approximant: str,
              true_params: TrueParams, task_meta: TaskMetadata, arrays: dict):
    # data/{approximant}/{task_id}/
    task_dir = os.path.join(base_outdir, approximant, task_meta.task_id)
    os.makedirs(task_dir, exist_ok=True)

    for key in ["strain_H1", "strain_L1", "psd_H1", "psd_L1", "psd_freqs", "times"]:
        np.save(os.path.join(task_dir, f"{key}.npy"), arrays[key])

    with open(os.path.join(task_dir, "task.json"), "w") as f:
        json.dump(asdict(task_meta), f, indent=2)

    with open(os.path.join(task_dir, "ground_truth.json"), "w") as f:
        json.dump(asdict(true_params), f, indent=2)

    print(f"  {task_meta.task_id}  SNR≈{true_params.network_snr:.1f}"
          f"  Mc={true_params.chirp_mass:.1f}  q={true_params.mass_ratio:.2f}")


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
def build_index(base_outdir: str, approximant: str):
    outdir = os.path.join(base_outdir, approximant)
    index  = {"approximant": approximant, "tasks": []}

    for task_id in sorted(os.listdir(outdir)):
        task_dir  = os.path.join(outdir, task_id)
        meta_path = os.path.join(task_dir, "task.json")
        gt_path   = os.path.join(task_dir, "ground_truth.json")
        if not os.path.isdir(task_dir) or not os.path.exists(meta_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        index["tasks"].append({
            "task_id": task_id,
            "tier":    gt["tier"],
            "path":    os.path.join(approximant, task_id),
        })

    index["total"]   = len(index["tasks"])
    index["by_tier"] = {
        t: len([x for x in index["tasks"] if x["tier"] == t])
        for t in ["easy", "medium", "hard"]
    }

    with open(os.path.join(outdir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex → {outdir}/index.json  "
          f"total={index['total']}  "
          f"easy={index['by_tier']['easy']}  "
          f"medium={index['by_tier']['medium']}  "
          f"hard={index['by_tier']['hard']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate GW Merger Bench dataset")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--approximant", type=str, default="IMRPhenomD",
                        choices=["IMRPhenomD", "SEOBNRv4", "IMRPhenomXHM"])
    parser.add_argument("--outdir",      type=str, default="data",
                        help="Base output directory. Tasks saved to <outdir>/<approximant>/")
    args = parser.parse_args()

    if not PYCBC_AVAILABLE:
        print("ERROR: pycbc not installed.")
        return

    global APPROXIMANT
    APPROXIMANT = args.approximant
    print(f"Approximant: {APPROXIMANT}")
    print(f"Output:      {args.outdir}/{APPROXIMANT}/")

    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.outdir, APPROXIMANT), exist_ok=True)

    task_counter = 0
    for tier, cfg in DIFFICULTY_CONFIG.items():
        print(f"\n--- {cfg['n_tasks']} {tier.upper()} tasks ---")
        for i in range(cfg["n_tasks"]):
            task_id = f"{task_counter:03d}"
            try:
                true_params, task_meta, arrays = generate_one_event(
                    task_id=task_id, tier=tier, cfg=cfg, rng=rng, np_rng=np_rng,
                )
                save_task(args.outdir, APPROXIMANT, true_params, task_meta, arrays)
                task_counter += 1
            except Exception as e:
                print(f"  ERROR on {task_id}: {e}")

    build_index(args.outdir, APPROXIMANT)
    print(f"\nDone. {task_counter} tasks in {args.outdir}/{APPROXIMANT}/")


if __name__ == "__main__":
    main()