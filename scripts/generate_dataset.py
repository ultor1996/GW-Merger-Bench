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
      task.json           — task metadata (public, no true params, no tier)
      ground_truth.json   — true parameters + tier (hidden from agent)

times.npy is NOT saved — sample_rate is in task.json and sufficient.
All tasks live in one flat folder under the approximant subfolder.
Tier is stored only in ground_truth.json — never given to the agent.

Usage:
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --seed 42 --approximant IMRPhenomD
"""

import argparse
import json
import os
import random
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
import sys
import logging 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    CHIRP_MASS_TOL_FRAC, MASS_RATIO_TOL_ABS, SNR_TOL_FRAC,
    SAMPLE_RATE, SEGMENT_DURATION, F_LOWER, APPROXIMANT,COA_TIME_FRAC,NOISE_SEED_STRATEGY, DISTANCE_RANGE_MPC,  OVERLAP_THRESHOLD
)
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.psd import aLIGOZeroDetHighPower
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    print("WARNING: pycbc not found. Install with: pip install pycbc")
from scipy.signal import hilbert


DIFFICULTY_CONFIG = {
    "easy": {
        "n_tasks":               5,
        "network_snr_range":     (20.0, 35.0),
        "total_mass_range":      (40.0, 73.0),
        "mass_ratio_range":      (0.7, 1.0),
        "spin_magnitude_range":  (0.0, 0.0), # zero spin  and inclination controlled benchmark
        "inclination_range":     (0.0, 0.0),
        "difficulty_score_range":(1, 3),
    },
    "medium": {
        "n_tasks":               5,
        "network_snr_range":     (12.0, 20.0),
        "total_mass_range":      (25.0, 73.0),
        "mass_ratio_range":      (0.4, 0.9),
        "spin_magnitude_range":  (0.0, 0.0),
        "inclination_range":     (0.0, 0.0),
        "difficulty_score_range":(4, 7),
    },
    "hard": {
        "n_tasks":               5,
        "network_snr_range":     (8.0, 12.0),
        "total_mass_range":      (10.0, 73.0),
        "mass_ratio_range":      (0.1, 0.6),
        "spin_magnitude_range":  (0.0, 0.0),
        "inclination_range":     (0.0, 0.0),
        "difficulty_score_range":(8, 10),
    },
}



@dataclass
class TrueParams:
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
    task_id:           str
    description:       str
    sample_rate:       int
    segment_duration:  float
    f_lower:           float
    detectors:         list
    approximant_hint:  str
    submission_format: dict
    data_files:        dict
    given_parameters:  dict   

def _measure_freq_evolution(hp_arr, dt, f_lower, f_ref=100.0, window_samples=50):
    """
    Measure instantaneous GW frequency near f_ref using the Hilbert transform,
    then estimate df/dt via a least-squares linear fit over a window of
    samples around that point (far more robust than a 4-point finite difference).
    Returns (f_measured, dfdt_measured) or (None, None) if it can't be measured.
    """
    analytic_signal = hilbert(hp_arr)
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) / (2.0 * np.pi * dt)

    valid = np.abs(hp_arr[:-1]) > (0.01 * np.max(np.abs(hp_arr)))
    inst_freq_valid = inst_freq[valid]
    valid_indices = np.where(valid)[0]  # original-array indices corresponding to inst_freq_valid

    if len(inst_freq_valid) < window_samples * 2:
        return None, None

    idx = np.argmin(np.abs(inst_freq_valid - f_ref))

    half_win = window_samples // 2
    lo = idx - half_win
    hi = idx + half_win
    if lo < 0 or hi >= len(inst_freq_valid):
        return None, None

    # Time axis for the window (using actual sample spacing)
    t_window = valid_indices[lo:hi] * dt
    f_window = inst_freq_valid[lo:hi]

    # Least-squares linear fit: f(t) ≈ f0 + dfdt * t
    A = np.vstack([t_window, np.ones_like(t_window)]).T
    dfdt_fit, f0_fit = np.linalg.lstsq(A, f_window, rcond=None)[0]

    f_measured = float(np.mean(f_window))   # average frequency across the window
    dfdt_measured = float(dfdt_fit)

    return f_measured, dfdt_measured



def _chirp_mass_from_dfdt(f, dfdt):
    G_over_c3 = 4.925491025543576e-06
    Mchirp_s  = (5.0 / 96.0 * np.pi ** (-8.0/3.0) *
                 f ** (-11.0/3.0) * dfdt) ** (3.0/5.0)
    return Mchirp_s / G_over_c3

def _measure_freq_evolution_robust(hp_arr, dt, f_lower, isco_freq, chirp_mass, max_tries=5):
    fractions = [0.15, 0.2, 0.25, 0.3, 0.4]

    for frac in fractions[:max_tries]:
        f_ref_try = min(100.0, frac * isco_freq)
        f_meas, dfdt_meas = _measure_freq_evolution(hp_arr, dt, f_lower, f_ref=f_ref_try, window_samples=50)

        if f_meas is None or dfdt_meas is None or dfdt_meas <= 0:
            continue

        candidate_mc = _chirp_mass_from_dfdt(f_meas, dfdt_meas)
        rel_diff = abs(candidate_mc - chirp_mass) / chirp_mass

        if rel_diff < 0.10:   # tighten acceptance to your target 10% threshold
            return f_meas, dfdt_meas

    return None, None

def _colored_noise(psd_vals, psd_freqs, n_samples, sample_rate, seed):
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

def generate_one_event(task_id, tier, cfg, rng, np_rng):
    """Generate a single synthetic BBH event with known parameters."""

    target_snr  = rng.uniform(*cfg["network_snr_range"])
    inclination = rng.uniform(*cfg["inclination_range"])
    spin_mag    = rng.uniform(*cfg["spin_magnitude_range"])
    ra          = rng.uniform(0, 2 * np.pi)
    dec         = rng.uniform(-np.pi / 2, np.pi / 2)
    polarisation= rng.uniform(0, np.pi)

    if COA_TIME_FRAC is None:
        coa_time_offset = rng.uniform(6.0, SEGMENT_DURATION - 2.0)
    else:
        coa_time_offset = SEGMENT_DURATION * COA_TIME_FRAC

    dt        = 1.0 / SAMPLE_RATE
    n_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    flen      = n_samples // 2 + 1
    delta_f   = 1.0 / SEGMENT_DURATION

    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_L1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    max_attempts = 20   # bumped up since we're now rejecting more cases
    for attempt in range(max_attempts):

        total_mass = rng.uniform(*cfg["total_mass_range"])
        mass_ratio = rng.uniform(*cfg["mass_ratio_range"])
        m1         = total_mass / (1.0 + mass_ratio)
        m2         = mass_ratio * m1
        spin1z     = spin_mag * rng.choice([-1, 1])
        spin2z     = spin_mag * rng.choice([-1, 1])
        chirp_mass = (m1 * m2)**(3.0/5.0) / (m1 + m2)**(1.0/5.0)

        # ── NEW: reject masses whose ISCO frequency leaves too little inspiral room ──
        isco_freq_check = 4397.9 / (m1 + m2)
        if isco_freq_check < F_LOWER * 3:
            logging.warning(
                f"Task {task_id} attempt {attempt+1}: ISCO freq {isco_freq_check:.1f} Hz "
                f"too close to F_LOWER={F_LOWER} Hz, resampling"
            )
            continue

        # Reference waveform at 100 Mpc to measure SNR scaling
        try:
            hp_ref, hc_ref = get_td_waveform(
                approximant=APPROXIMANT,
                mass1=m1, mass2=m2,
                spin1z=spin1z, spin2z=spin2z,
                inclination=inclination, coa_phase=0.0,
                delta_t=dt, f_lower=F_LOWER, distance=100.0,
            )
        except Exception as e:
            logging.warning(f"Task {task_id} attempt {attempt+1}: waveform failed: {e}")
            continue

        sig_arr_ref  = np.zeros(n_samples)
        coa_idx      = int(coa_time_offset * SAMPLE_RATE)
        hp_arr_ref   = np.array(hp_ref)
        peak_idx_ref = int(np.argmax(np.abs(hp_arr_ref)))
        src_start    = max(0, peak_idx_ref - coa_idx)
        dst_start    = max(0, coa_idx - peak_idx_ref)
        copy_len     = min(len(hp_arr_ref) - src_start, n_samples - dst_start)
        if copy_len > 0:
            sig_arr_ref[dst_start:dst_start + copy_len] = hp_arr_ref[src_start:src_start + copy_len]

        psd_freqs_arr = np.linspace(0, SAMPLE_RATE / 2, flen)
        psd_vals_H1   = np.array(psd_H1)
        psd_vals_L1   = np.array(psd_L1)

        det_H1 = Detector("H1")
        det_L1 = Detector("L1")
        ref_gps = 1264316116.0
        gps_coa = ref_gps + coa_time_offset
        fp_H1, fc_H1 = det_H1.antenna_pattern(ra, dec, polarisation, gps_coa)
        fp_L1, fc_L1 = det_L1.antenna_pattern(ra, dec, polarisation, gps_coa)

        try:
            hc_ref_arr = np.array(hc_ref)
            sig_hc_ref = np.zeros(n_samples)
            if copy_len > 0:
                sig_hc_ref[dst_start:dst_start + copy_len] = hc_ref_arr[src_start:src_start + copy_len]

            sig_H1_ref = fp_H1 * sig_arr_ref + fc_H1 * sig_hc_ref
            sig_L1_ref = fp_L1 * sig_arr_ref + fc_L1 * sig_hc_ref

            def _compute_snr(sig, psd_vals):
                sig_f   = np.fft.rfft(sig) * dt
                freqs   = np.fft.rfftfreq(n_samples, d=dt)
                psd_i   = np.interp(freqs, psd_freqs_arr, psd_vals, left=1e-40, right=1e-40)
                psd_i   = np.where(psd_i > 0, psd_i, 1e-40)
                integrand = (np.abs(sig_f)**2 / psd_i)
                mask    = freqs >= F_LOWER
                snr_sq  = 4.0 * np.sum(integrand[mask]) * delta_f
                return float(np.sqrt(max(snr_sq, 0.0)))

            snr_H1_ref      = _compute_snr(sig_H1_ref, psd_vals_H1)
            snr_L1_ref      = _compute_snr(sig_L1_ref, psd_vals_L1)
            network_snr_100 = float(np.sqrt(snr_H1_ref**2 + snr_L1_ref**2))
        except Exception as e:
            logging.warning(f"Task {task_id} attempt {attempt+1}: SNR computation failed: {e}")
            continue

        distance     = 100.0 * network_snr_100 / target_snr if network_snr_100 > 0 else 500.0
        d_min, d_max = DISTANCE_RANGE_MPC[tier]
        distance     = float(np.clip(distance, d_min, d_max))

        if network_snr_100 <= 0:
            logging.warning(f"Task {task_id} attempt {attempt+1}: SNR=0, resampling")
            continue

        # ── Generate final waveform at correct distance, right here inside the loop ──
        try:
            hp, hc = get_td_waveform(
                approximant=APPROXIMANT,
                mass1=m1, mass2=m2,
                spin1z=spin1z, spin2z=spin2z,
                inclination=inclination, coa_phase=0.0,
                delta_t=dt, f_lower=F_LOWER, distance=distance,
            )
        except Exception as e:
            logging.warning(f"Task {task_id} attempt {attempt+1}: final waveform failed: {e}")
            continue

        hp_arr = np.array(hp)
        hc_arr = np.array(hc)
        sig_hp = np.zeros(n_samples)
        sig_hc = np.zeros(n_samples)
        peak_idx  = int(np.argmax(np.abs(hp_arr)))
        src_start = max(0, peak_idx - coa_idx)
        dst_start = max(0, coa_idx - peak_idx)
        copy_len  = min(len(hp_arr) - src_start, n_samples - dst_start)
        if copy_len > 0:
            sig_hp[dst_start:dst_start + copy_len] = hp_arr[src_start:src_start + copy_len]
            sig_hc[dst_start:dst_start + copy_len] = hc_arr[src_start:src_start + copy_len]
        sig_H1 = fp_H1 * sig_hp + fc_H1 * sig_hc
        sig_L1 = fp_L1 * sig_hp + fc_L1 * sig_hc

        snr_H1      = _compute_snr(sig_H1, psd_vals_H1)
        snr_L1      = _compute_snr(sig_L1, psd_vals_L1)
        network_snr = float(np.sqrt(snr_H1**2 + snr_L1**2))

        isco_freq = 4397.9 / (m1 + m2)
        f_meas, dfdt_meas = _measure_freq_evolution_robust(
            hp_arr, dt, F_LOWER, isco_freq, chirp_mass, max_tries=10
        )

        # ── NEW: if freq-evo measurement fails, resample masses instead of accepting null ──
        if f_meas is None or dfdt_meas is None or dfdt_meas <= 0:
            logging.warning(
                f"Task {task_id} attempt {attempt+1}: freq-evo measurement failed "
                f"(M_total={m1+m2:.1f}, ISCO={isco_freq:.1f} Hz), resampling masses"
            )
            continue

        chirp_mass_from_freq_evo = _chirp_mass_from_dfdt(f_meas, dfdt_meas)

        # Everything succeeded — break out with all values set
        break

    else:
        raise RuntimeError(
            f"Task {task_id}: could not find valid parameters with a measurable "
            f"freq-evolution chirp mass after {max_attempts} attempts — "
            f"consider raising total_mass_range lower bound or lowering F_LOWER for tier '{tier}'"
        )

    # ── Noise ─────────────────────────────────────────────────────────
    def _noise_seed(task_id, detector):
        if NOISE_SEED_STRATEGY == "task_id":
            h = hashlib.sha256((task_id + detector).encode()).hexdigest()
            return int(h, 16) % (2 ** 31)
        elif NOISE_SEED_STRATEGY == "random":
            return np_rng.integers(0, 2 ** 31)
        else:
            return int(NOISE_SEED_STRATEGY)

    noise_H1 = _colored_noise(psd_vals_H1, psd_freqs_arr, n_samples, SAMPLE_RATE, seed=_noise_seed(task_id, "H1"))
    noise_L1 = _colored_noise(psd_vals_L1, psd_freqs_arr, n_samples, SAMPLE_RATE, seed=_noise_seed(task_id, "L1"))

    strain_H1 = noise_H1 + sig_H1
    strain_L1 = noise_L1 + sig_L1

    peak_freq_hz = min(isco_freq, SAMPLE_RATE / 2)

    difficulty_lo, difficulty_hi = cfg["difficulty_score_range"]
    difficulty_score = rng.randint(difficulty_lo, difficulty_hi)

    true_params = TrueParams(
        task_id=task_id, tier=tier, difficulty_score=difficulty_score,
        mass1=round(m1, 4), mass2=round(m2, 4),
        chirp_mass=round(chirp_mass, 4), mass_ratio=round(mass_ratio, 4),
        spin1z=round(spin1z, 4), spin2z=round(spin2z, 4),
        distance=round(distance, 2), inclination=round(inclination, 4),
        ra=round(ra, 4), dec=round(dec, 4), polarisation=round(polarisation, 4),
        coalescence_time=round(coa_time_offset, 4),
        network_snr=round(network_snr, 3),
        chirp_mass_from_freq_evo=round(chirp_mass_from_freq_evo, 4),  # always a real number now
        peak_frequency_hz=round(peak_freq_hz, 2),
        optimal_snr_H1=round(snr_H1, 3), optimal_snr_L1=round(snr_L1, 3),
        merger_type="BBH", approximant=APPROXIMANT,
        chirp_mass_tol_frac=CHIRP_MASS_TOL_FRAC,
        mass_ratio_tol_abs=MASS_RATIO_TOL_ABS,
        snr_tol_frac=SNR_TOL_FRAC,
    )
    task_meta = TaskMetadata(
    task_id=task_id,
    description=(
        "A gravitational-wave strain signal has been recorded by the H1 and L1 "
        "detectors. The segment is 16s long at 2048 Hz. The sky location "
        "(ra, dec, polarisation) is given in given_parameters. Estimate the chirp mass and coalescence time "
        "of the signal."
    ),
    sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, f_lower=F_LOWER,
    detectors=["H1", "L1"], approximant_hint=APPROXIMANT,
    submission_format={
        "chirp_mass_Msun":    "float",
        "coalescence_time_s": "float — merger time within the segment",
    },
    data_files={
        "strain_H1": "strain_H1.npy", "strain_L1": "strain_L1.npy",
        "psd_H1":    "psd_H1.npy",    "psd_L1":    "psd_L1.npy",
        "psd_freqs": "psd_freqs.npy",
    },
    given_parameters={
        "ra": round(float(ra), 6),
        "dec": round(float(dec), 6),
        "polarisation": round(float(polarisation), 6),
    },
)

    arrays = {
        "strain_H1":  strain_H1.astype(np.float64),
        "strain_L1":  strain_L1.astype(np.float64),
        "psd_H1":     psd_vals_H1.astype(np.float64),
        "psd_L1":     psd_vals_L1.astype(np.float64),
        "psd_freqs":  psd_freqs_arr.astype(np.float64),
    }

    return true_params, task_meta, arrays


def save_task(base_outdir, approximant, true_params, task_meta, arrays):
    task_dir = os.path.join(base_outdir, approximant, task_meta.task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Save only the 5 data files — no times.npy
    for key in ["strain_H1", "strain_L1", "psd_H1", "psd_L1", "psd_freqs"]:
        np.save(os.path.join(task_dir, f"{key}.npy"), arrays[key])

    with open(os.path.join(task_dir, "task.json"), "w") as f:
        json.dump(asdict(task_meta), f, indent=2)

    with open(os.path.join(task_dir, "ground_truth.json"), "w") as f:
        json.dump(asdict(true_params), f, indent=2)

    print(f"  {task_meta.task_id}  SNR≈{true_params.network_snr:.1f}"
          f"  Mc={true_params.chirp_mass:.1f}  q={true_params.mass_ratio:.2f}")


def build_index(base_outdir, approximant):
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


def main():
    parser = argparse.ArgumentParser(description="Generate GW Merger Bench dataset")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--approximant", type=str, default="IMRPhenomD",
                        choices=["IMRPhenomD", "SEOBNRv4", "IMRPhenomXHM"])
    parser.add_argument("--outdir",      type=str, default="data")
    args = parser.parse_args()

    if not PYCBC_AVAILABLE:
        print("ERROR: pycbc not installed.")
        return

    global APPROXIMANT
    APPROXIMANT = args.approximant
    print(f"Approximant: {APPROXIMANT}")
    print(f"Output:      {args.outdir}/{APPROXIMANT}/")
    print(f"Note: times.npy will NOT be saved — sample_rate in task.json is sufficient.")

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