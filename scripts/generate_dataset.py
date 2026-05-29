"""
GW Merger Bench — Dataset Generator
Generates 60 synthetic BBH injection tasks (20 easy / 20 medium / 20 hard)
mirroring Stargazer's synthetic task generation philosophy.

Each task is saved as:
  data/synthetic/{tier}/{task_id}/
      strain_H1.npy       — detector strain time series (H1)
      strain_L1.npy       — detector strain time series (L1)
      psd_H1.npy          — noise PSD used (H1)
      psd_L1.npy          — noise PSD used (L1)
      times.npy           — GPS-relative time array
      task.json           — task metadata (public, no true params)
      ground_truth.json   — true parameters (hidden from agent)

Usage:
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --seed 99 --outdir data/synthetic
"""

import argparse
import json
import os
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple

# ---------------------------------------------------------------------------
# Lazy PyCBC imports so the script fails gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.noise import noise_from_string
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import matched_filter, sigma
    from pycbc.types import TimeSeries, FrequencySeries
    import pycbc.psd as pycbc_psd
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    print("WARNING: pycbc not found. Install with: pip install pycbc")


# ---------------------------------------------------------------------------
# Difficulty configuration
# Mirrors Stargazer's six physical factors mapped to GW parameters
# ---------------------------------------------------------------------------
DIFFICULTY_CONFIG = {
    "easy": {
        "n_tasks": 20,
        # SNR range: clearly detectable
        "network_snr_range": (20.0, 35.0),
        # Total mass: intermediate — signal is short but clearly visible
        "total_mass_range": (40.0, 80.0),
        # Mass ratio q = m2/m1, q in (0,1]
        "mass_ratio_range": (0.7, 1.0),
        # Aligned spin magnitude
        "spin_magnitude_range": (0.0, 0.1),
        # Inclination: face-on systems are loudest
        "inclination_range": (0.0, 0.3),
        # Distance derived from SNR — set implicitly
        "noise_type": "gaussian",
        "difficulty_score_range": (1, 3),
    },
    "medium": {
        "n_tasks": 20,
        "network_snr_range": (12.0, 20.0),
        "total_mass_range": (25.0, 120.0),
        "mass_ratio_range": (0.4, 0.9),
        "spin_magnitude_range": (0.0, 0.5),
        "inclination_range": (0.0, 1.0),
        "noise_type": "gaussian",
        "difficulty_score_range": (4, 7),
    },
    "hard": {
        "n_tasks": 20,
        "network_snr_range": (8.0, 12.0),
        "total_mass_range": (10.0, 200.0),
        "mass_ratio_range": (0.1, 0.6),
        "spin_magnitude_range": (0.3, 0.9),
        "inclination_range": (0.5, np.pi / 2),
        "noise_type": "gaussian",
        "difficulty_score_range": (8, 10),
    },
}

SAMPLE_RATE = 2048       # Hz — sufficient for BBH up to ~500 Hz
SEGMENT_DURATION = 16    # seconds of data given to agent
F_LOWER = 20.0           # Hz — low-frequency cutoff
APPROXIMANT = "IMRPhenomD"


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrueParams:
    """Ground truth parameters — never shown to the agent."""
    task_id: str
    tier: str
    difficulty_score: int

    # Intrinsic
    mass1: float          # solar masses (heavier)
    mass2: float          # solar masses (lighter)
    chirp_mass: float     # solar masses
    mass_ratio: float     # m2/m1, in (0, 1]
    spin1z: float         # dimensionless aligned spin on mass1
    spin2z: float         # dimensionless aligned spin on mass2

    # Geometric / extrinsic
    distance: float       # Mpc
    inclination: float    # radians
    ra: float             # radians
    dec: float            # radians
    polarisation: float   # radians
    coalescence_time: float  # seconds into segment (GPS offset)

    # Computed SNR
    network_snr: float

    # Anchor ground truths (recomputable by evaluator)
    chirp_mass_from_freq_evo: float   # should match chirp_mass
    peak_frequency_hz: float          # frequency at max amplitude
    optimal_snr_H1: float
    optimal_snr_L1: float
    merger_type: str                  # "BBH" for all synthetic tasks
    approximant: str                  # waveform model used to generate data

    # Evaluation thresholds
    chirp_mass_tol_frac: float        # fractional tolerance for AP3
    mass_ratio_tol_abs: float         # absolute tolerance for AP4
    snr_tol_frac: float               # fractional tolerance for AP2


@dataclass
class TaskMetadata:
    """Public task description given to the agent."""
    task_id: str
    tier: str
    difficulty_score: int
    description: str
    sample_rate: int
    segment_duration: float
    f_lower: float
    detectors: list
    approximant_hint: str
    submission_format: dict


# ---------------------------------------------------------------------------
# Chirp mass formula
# ---------------------------------------------------------------------------
def chirp_mass(m1: float, m2: float) -> float:
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def mass_ratio(m1: float, m2: float) -> float:
    """Returns q = m2/m1 with m1 >= m2, so q in (0,1]."""
    return min(m1, m2) / max(m1, m2)


# ---------------------------------------------------------------------------
# Analytic chirp-mass from frequency evolution (anchor ground truth)
# ---------------------------------------------------------------------------
def chirp_mass_from_dfdt(f: float, dfdt: float) -> float:
    """
    𝓜 = c³/G * (5/96 * π^(-8/3) * f^(-11/3) * df/dt)^(3/5)
    In solar mass units with G/c³ = 4.926e-6 s/M_sun.
    """
    G_over_c3 = 4.925491025543576e-06  # seconds per solar mass
    Mchirp_s = (5.0 / 96.0 * np.pi ** (-8.0 / 3.0) * f ** (-11.0 / 3.0) * dfdt) ** (3.0 / 5.0)
    return Mchirp_s / G_over_c3


# ---------------------------------------------------------------------------
# Main generation function for one event
# ---------------------------------------------------------------------------
def generate_one_event(
    task_id: str,
    tier: str,
    cfg: dict,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[TrueParams, TaskMetadata, dict]:
    """
    Returns (true_params, task_metadata, arrays_dict).
    arrays_dict contains: strain_H1, strain_L1, psd_H1, psd_L1, times
    """
    if not PYCBC_AVAILABLE:
        raise RuntimeError("pycbc is required to generate data.")

    # --- Sample parameters ---
    total_mass = rng.uniform(*cfg["total_mass_range"])
    q = rng.uniform(*cfg["mass_ratio_range"])
    m1 = total_mass / (1.0 + q)
    m2 = total_mass * q / (1.0 + q)
    if m1 < m2:
        m1, m2 = m2, m1

    spin1z = rng.uniform(*cfg["spin_magnitude_range"]) * rng.choice([-1, 1])
    spin2z = rng.uniform(*cfg["spin_magnitude_range"]) * rng.choice([-1, 1])
    inclination = rng.uniform(*cfg["inclination_range"])
    ra = rng.uniform(0, 2 * np.pi)
    dec = rng.uniform(-np.pi / 2, np.pi / 2)
    polarisation = rng.uniform(0, np.pi)
    coa_phase = rng.uniform(0, 2 * np.pi)

    # Coalescence placed at 2/3 through segment so inspiral is visible
    coa_time_offset = SEGMENT_DURATION * 0.67

    dt = 1.0 / SAMPLE_RATE
    flen = int(SEGMENT_DURATION * SAMPLE_RATE / 2) + 1

    # --- Generate waveform ---
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=m1,
        mass2=m2,
        spin1z=spin1z,
        spin2z=spin2z,
        inclination=inclination,
        coa_phase=coa_phase,
        delta_t=dt,
        f_lower=F_LOWER,
        distance=100.0,   # placeholder distance, rescaled by SNR below
    )

    # --- Build PSD ---
    delta_f = 1.0 / SEGMENT_DURATION
    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_L1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    # --- Project onto detectors ---
    det_H1 = Detector("H1")
    det_L1 = Detector("L1")

    # Use a fixed reference GPS time (O3-era)
    ref_gps = 1264316116.0
    gps_coa = ref_gps + coa_time_offset

    fp_H1, fc_H1 = det_H1.antenna_pattern(ra, dec, polarisation, gps_coa)
    fp_L1, fc_L1 = det_L1.antenna_pattern(ra, dec, polarisation, gps_coa)

    # Resize waveform to segment length
    n_samples = int(SEGMENT_DURATION * SAMPLE_RATE)

    def project_and_resize(hp, hc, fp, fc, n):
        sig = fp * hp + fc * hc
        # align coalescence to desired offset
        sig_arr = np.zeros(n)
        coa_idx = int(coa_time_offset * SAMPLE_RATE)
        # hp ends at coalescence in pycbc convention
        end_idx = min(coa_idx, len(sig))
        start_src = len(sig) - end_idx
        sig_arr[coa_idx - end_idx: coa_idx] = np.array(sig)[start_src:]
        return sig_arr

    sig_H1 = project_and_resize(hp, hc, fp_H1, fc_H1, n_samples)
    sig_L1 = project_and_resize(hp, hc, fp_L1, fc_L1, n_samples)

    # --- Compute optimal SNR at distance=100 Mpc ---
    hp_ts = TimeSeries(sig_H1, delta_t=dt)
    hc_ts = TimeSeries(sig_L1, delta_t=dt)

    # Compute sigma (optimal SNR) for normalisation
    try:
        sig_H1_ts = TimeSeries(sig_H1, delta_t=dt)
        opt_snr_H1_100 = float(sigma(sig_H1_ts, psd=psd_H1, low_frequency_cutoff=F_LOWER))
        opt_snr_L1_100 = float(sigma(hc_ts, psd=psd_L1, low_frequency_cutoff=F_LOWER))
    except Exception:
        opt_snr_H1_100 = 10.0
        opt_snr_L1_100 = 10.0

    network_snr_100 = np.sqrt(opt_snr_H1_100 ** 2 + opt_snr_L1_100 ** 2)

    # Rescale distance so network SNR matches target
    target_snr = rng.uniform(*cfg["network_snr_range"])
    if network_snr_100 > 0:
        distance = 100.0 * network_snr_100 / target_snr
    else:
        distance = 500.0

    scale = 100.0 / distance
    sig_H1 *= scale
    sig_L1 *= scale
    opt_snr_H1 = opt_snr_H1_100 * scale
    opt_snr_L1 = opt_snr_L1_100 * scale
    network_snr = np.sqrt(opt_snr_H1 ** 2 + opt_snr_L1 ** 2)

    # --- Generate Gaussian noise and inject ---
    noise_H1 = noise_from_string(
        "aLIGOZeroDetHighPower",
        length=n_samples,
        delta_t=dt,
        low_frequency_cutoff=F_LOWER,
        seed=abs(hash(task_id + "H1")) % (2 ** 31),
    )
    noise_L1 = noise_from_string(
        "aLIGOZeroDetHighPower",
        length=n_samples,
        delta_t=dt,
        low_frequency_cutoff=F_LOWER,
        seed=abs(hash(task_id + "L1")) % (2 ** 31),
    )

    noise_arr_H1 = np.array(noise_H1)[:n_samples]
    noise_arr_L1 = np.array(noise_L1)[:n_samples]

    strain_H1 = noise_arr_H1 + sig_H1
    strain_L1 = noise_arr_L1 + sig_L1

    # --- Compute anchor values ---
    Mc = chirp_mass(m1, m2)

    # Peak GW frequency ~ frequency at max amplitude of h(t)
    # For IMRPhenomD this is approximately the ISCO frequency
    isco_freq = 4400.0 / (m1 + m2)  # Hz, approximate

    # Chirp mass from frequency evolution (analytic, at f=100 Hz)
    f_ref = 100.0
    # df/dt at f_ref for a circular inspiral
    G_over_c3 = 4.925491025543576e-06
    Mc_s = Mc * G_over_c3
    dfdt_ref = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * Mc_s ** (5.0 / 3.0) * f_ref ** (11.0 / 3.0)
    Mc_anchor = chirp_mass_from_dfdt(f_ref, dfdt_ref)  # should == Mc

    # Difficulty score within tier range
    lo, hi = cfg["difficulty_score_range"]
    diff_score = rng.randint(lo, hi)

    # --- Build output objects ---
    true_params = TrueParams(
        task_id=task_id,
        tier=tier,
        difficulty_score=diff_score,
        mass1=round(m1, 4),
        mass2=round(m2, 4),
        chirp_mass=round(Mc, 4),
        mass_ratio=round(q, 4),
        spin1z=round(spin1z, 4),
        spin2z=round(spin2z, 4),
        distance=round(distance, 2),
        inclination=round(inclination, 4),
        ra=round(ra, 4),
        dec=round(dec, 4),
        polarisation=round(polarisation, 4),
        coalescence_time=round(coa_time_offset, 4),
        network_snr=round(network_snr, 3),
        chirp_mass_from_freq_evo=round(Mc_anchor, 4),
        peak_frequency_hz=round(isco_freq, 2),
        optimal_snr_H1=round(opt_snr_H1, 3),
        optimal_snr_L1=round(opt_snr_L1, 3),
        merger_type="BBH",
        approximant=APPROXIMANT,
        chirp_mass_tol_frac=0.05,
        mass_ratio_tol_abs=0.15,
        snr_tol_frac=0.20,
    )

    task_meta = TaskMetadata(
        task_id=task_id,
        tier=tier,
        difficulty_score=diff_score,
        description=(
            f"A gravitational-wave strain signal has been recorded by the H1 and L1 detectors. "
            f"The segment is {SEGMENT_DURATION}s long at {SAMPLE_RATE} Hz. "
            f"Your task is to detect the signal, estimate the chirp mass, component masses, "
            f"mass ratio, spins, distance, and sky location, and classify the merger type. "
            f"Submit your estimates using the submit_action tool. "
            f"You may revise your submission based on evaluator feedback."
        ),
        sample_rate=SAMPLE_RATE,
        segment_duration=SEGMENT_DURATION,
        f_lower=F_LOWER,
        detectors=["H1", "L1"],
        approximant_hint=APPROXIMANT,
        submission_format={
            "chirp_mass_Msun": "float",
            "mass1_Msun": "float",
            "mass2_Msun": "float",
            "mass_ratio": "float — m2/m1, in (0,1]",
            "spin1z": "float — dimensionless [-1, 1]",
            "spin2z": "float — dimensionless [-1, 1]",
            "distance_Mpc": "float",
            "inclination_rad": "float",
            "ra_rad": "float",
            "dec_rad": "float",
            "network_snr": "float — your estimated SNR",
            "merger_type": "str — one of BBH / BNS / NSBH",
            "confidence": "float — your confidence in [0, 1]",
        },
    )

    times = np.linspace(0, SEGMENT_DURATION, n_samples, endpoint=False)
    psd_freqs = np.linspace(0, SAMPLE_RATE / 2, flen)
    psd_vals_H1 = np.array(psd_H1)
    psd_vals_L1 = np.array(psd_L1)

    arrays = {
        "strain_H1": strain_H1,
        "strain_L1": strain_L1,
        "psd_H1": psd_vals_H1,
        "psd_L1": psd_vals_L1,
        "psd_freqs": psd_freqs,
        "times": times,
    }

    return true_params, task_meta, arrays


# ---------------------------------------------------------------------------
# Save one task to disk
# ---------------------------------------------------------------------------
def save_task(outdir: str, true_params: TrueParams, task_meta: TaskMetadata, arrays: dict):
    task_dir = os.path.join(outdir, task_meta.tier, task_meta.task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Public files
    np.save(os.path.join(task_dir, "strain_H1.npy"), arrays["strain_H1"])
    np.save(os.path.join(task_dir, "strain_L1.npy"), arrays["strain_L1"])
    np.save(os.path.join(task_dir, "psd_H1.npy"), arrays["psd_H1"])
    np.save(os.path.join(task_dir, "psd_L1.npy"), arrays["psd_L1"])
    np.save(os.path.join(task_dir, "psd_freqs.npy"), arrays["psd_freqs"])
    np.save(os.path.join(task_dir, "times.npy"), arrays["times"])

    with open(os.path.join(task_dir, "task.json"), "w") as f:
        json.dump(asdict(task_meta), f, indent=2)

    # Hidden ground truth
    with open(os.path.join(task_dir, "ground_truth.json"), "w") as f:
        json.dump(asdict(true_params), f, indent=2)

    print(f"  Saved {task_meta.task_id}  SNR≈{true_params.network_snr:.1f}"
          f"  𝓜={true_params.chirp_mass:.1f} M☉"
          f"  q={true_params.mass_ratio:.2f}"
          f"  spin1z={true_params.spin1z:.2f}")


# ---------------------------------------------------------------------------
# Build task index
# ---------------------------------------------------------------------------
def build_index(outdir: str):
    index = {"tasks": []}
    for tier in ["easy", "medium", "hard"]:
        tier_dir = os.path.join(outdir, tier)
        if not os.path.isdir(tier_dir):
            continue
        for task_id in sorted(os.listdir(tier_dir)):
            meta_path = os.path.join(tier_dir, task_id, "task.json")
            gt_path = os.path.join(tier_dir, task_id, "ground_truth.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            entry = {
                "task_id": task_id,
                "tier": tier,
                "difficulty_score": meta["difficulty_score"],
                "path": os.path.join(tier, task_id),
            }
            index["tasks"].append(entry)

    index["total"] = len(index["tasks"])
    index["by_tier"] = {
        t: len([x for x in index["tasks"] if x["tier"] == t])
        for t in ["easy", "medium", "hard"]
    }

    with open(os.path.join(outdir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex written: {index['total']} tasks  "
          f"easy={index['by_tier']['easy']}  "
          f"medium={index['by_tier']['medium']}  "
          f"hard={index['by_tier']['hard']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate GW Merger Bench dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--approximant", type=str, default="IMRPhenomD",
                        choices=["IMRPhenomD", "SEOBNRv4", "IMRPhenomXHM"],
                        help="Waveform approximant used to generate data")
    parser.add_argument("--outdir", type=str, default="data/synthetic",
                        help="Output directory")
    args = parser.parse_args()

    if not PYCBC_AVAILABLE:
        print("ERROR: pycbc not installed. Run: pip install pycbc")
        return

    # Override module-level APPROXIMANT with CLI choice
    global APPROXIMANT
    APPROXIMANT = args.approximant
    print(f"Using approximant: {APPROXIMANT}")

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    task_counter = 0
    for tier, cfg in DIFFICULTY_CONFIG.items():
        print(f"\n--- Generating {cfg['n_tasks']} {tier.upper()} tasks ---")
        for i in range(cfg["n_tasks"]):
            task_id = f"synthetic_{tier}_{i+1:03d}"
            try:
                true_params, task_meta, arrays = generate_one_event(
                    task_id=task_id,
                    tier=tier,
                    cfg=cfg,
                    rng=rng,
                    np_rng=np_rng,
                )
                save_task(args.outdir, true_params, task_meta, arrays)
                task_counter += 1
            except Exception as e:
                print(f"  ERROR on {task_id}: {e}")

    build_index(args.outdir)
    print(f"\nDone. Generated {task_counter} tasks in {args.outdir}/")


if __name__ == "__main__":
    main()