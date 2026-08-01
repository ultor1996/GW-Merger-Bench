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


# DIFFICULTY_CONFIG = {
#     "easy": {
#         "n_tasks":               5,
#         "network_snr_range":     (20.0, 35.0),
#         "total_mass_range":      (40.0, 73.0),
#         "mass_ratio_range":      (0.7, 1.0),
#         "spin_magnitude_range":  (0.0, 0.0), # zero spin  and inclination controlled benchmark
#         "inclination_range":     (0.0, 0.0),
#         "difficulty_score_range":(1, 3),
#     },
#     "medium": {
#         "n_tasks":               5,
#         "network_snr_range":     (12.0, 20.0),
#         "total_mass_range":      (25.0, 73.0),
#         "mass_ratio_range":      (0.4, 0.9),
#         "spin_magnitude_range":  (0.0, 0.0),
#         "inclination_range":     (0.0, 0.0),
#         "difficulty_score_range":(4, 7),
#     },
#     "hard": {
#         "n_tasks":               5,
#         "network_snr_range":     (8.0, 12.0),
#         "total_mass_range":      (10.0, 73.0),
#         "mass_ratio_range":      (0.1, 0.6),
#         "spin_magnitude_range":  (0.0, 0.0),
#         "inclination_range":     (0.0, 0.0),
#         "difficulty_score_range":(8, 10),
#     },
# }

DIFFICULTY_CONFIG = {
    "easy": {
        "n_tasks":               5,
        "network_snr_range":     (20.0, 35.0),
        "total_mass_range":      (40.0, 73.0),
        "mass_ratio_range":      (0.7, 1.0),
        "spin_magnitude_range":  (0.0, 0.2),   # was (0.0, 0.0)
        "inclination_range":     (0.0, 0.0),   # unchanged
        "difficulty_score_range":(1, 3),
    },
    "medium": {
        "n_tasks":               5,
        "network_snr_range":     (12.0, 20.0),
        "total_mass_range":      (25.0, 73.0),
        "mass_ratio_range":      (0.4, 0.9),
        "spin_magnitude_range":  (0.0, 0.5),   # was (0.0, 0.0)
        "inclination_range":     (0.0, 0.0),   # unchanged
        "difficulty_score_range":(4, 7),
    },
    "hard": {
        "n_tasks":               5,
        "network_snr_range":     (8.0, 12.0),
        "total_mass_range":      (10.0, 73.0),
        "mass_ratio_range":      (0.1, 0.6),
        "spin_magnitude_range":  (0.0, 0.8),   # was (0.0, 0.0)
        "inclination_range":     (0.0, 0.0),   # unchanged
        "difficulty_score_range":(8, 10),
    },
}

"""
Add this near the top of generate_dataset.py, right after DIFFICULTY_CONFIG.
Controls every realism factor independently. All defaults are "off" /
zero-effect, so existing behavior is unchanged until you flip something.
"""

REALISM_CONFIG = {
    "spectral_lines": {
        "enabled": False,
        # (frequency_hz, relative_power_factor) -- how many times louder
        # than the broadband PSD at that exact frequency. Mains harmonics
        # + a couple of fake "violin mode" style narrow features.
        "lines": [
            (60.0,   80.0),
            (120.0,  40.0),
            (180.0,  20.0),
            (500.5,  60.0),
            (1000.8, 30.0),
        ],
        "line_width_hz": 0.5,  # width of each Lorentzian bump
    },
    "calibration_error": {
        "enabled": False,
        "amplitude_frac_range": (0.0, 0.05),   # up to 5% amplitude distortion
        "phase_deg_range":      (0.0, 5.0),    # up to 5 degrees phase distortion
        "n_spline_nodes":       5,             # smooth spline control points across the band
    },
    "glitches": {
        "enabled": False,
        "probability":        0.3,             # fraction of tasks that get a glitch
        "snr_range":          (5.0, 15.0),     # glitch "loudness" in matched-filter-SNR-like units
        "duration_range_s":   (0.01, 0.1),
        "freq_range_hz":      (30.0, 500.0),
        "detector":           "random",        # "H1", "L1", or "random" (real glitches are per-detector)
    },
}

# Optional: scale realism factors up with difficulty tier, mirroring how
# real "hard" data segments tend to be messier. Multiplies the base
# REALISM_CONFIG values above. Set a tier to None to use the base config
# unscaled.
REALISM_TIER_SCALING = {
    "easy":   {"glitch_probability_mult": 0.0, "calibration_mult": 0.5},
    "medium": {"glitch_probability_mult": 1.0, "calibration_mult": 1.0},
    "hard":   {"glitch_probability_mult": 2.0, "calibration_mult": 1.5},
}

PRECESSION_CONFIG = {
    "enabled": False,
    "approximant": "IMRPhenomXPHM",   # overrides APPROXIMANT for injection when enabled
    "a_magnitude_range": (0.0, 0.8),
    "phi_12_range": (0.0, 6.283185307179586),   # 2*pi
    "phi_jl_range": (0.0, 6.283185307179586),
    # tilt_1/tilt_2/theta_jn are drawn isotropically (uniform in cos),
    # not from a range here -- see _sample_precessing_spins.
}
 
# --- CLI flags to add inside main()'s argparse block ---
"""
    parser.add_argument("--enable-spin", action="store_true",
                        help="Use nonzero spin_magnitude_range from DIFFICULTY_CONFIG "
                             "(edit the ranges there to control magnitude)")
    parser.add_argument("--enable-inclination", action="store_true",
                        help="Use nonzero inclination_range from DIFFICULTY_CONFIG")
    parser.add_argument("--enable-lines", action="store_true",
                        help="Add spectral lines (mains harmonics, violin-mode-like features) to the PSD")
    parser.add_argument("--enable-calibration-error", action="store_true",
                        help="Apply an unmodeled smooth calibration distortion to injected signals")
    parser.add_argument("--enable-glitches", action="store_true",
                        help="Randomly inject short non-Gaussian transients into the strain")
    parser.add_argument("--realism-seed", type=int, default=None,
                        help="Separate seed for realism-factor randomness (defaults to --seed)")
"""

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
    has_glitch:                bool = False
    glitch_detector:           str = None
    glitch_time_s:              float = None
    glitch_freq_hz:              float = None
    glitch_snr_like:              float = None
    spectral_lines_present:    bool = False
    calibration_error_present: bool = False
    is_precessing:          bool = False
    injection_approximant:  str = None
    a_1_magnitude:          float = None
    a_2_magnitude:          float = None
    tilt_1:                 float = 0.0
    tilt_2:                 float = 0.0
    phi_12:                 float = 0.0
    phi_jl:                 float = 0.0
    theta_jn_true:          float = 0.0
    spin1x:                 float = 0.0
    spin1y:                 float = 0.0
    spin2x:                 float = 0.0
    spin2y:                 float = 0.0


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
    sigma_f    = 0.5 * np.sqrt(psd_interp * sample_rate * n_samples)
    noise_f    = (rng_n.standard_normal(flen) +
                  1j * rng_n.standard_normal(flen)) * sigma_f
    noise_f[0]  = noise_f[0].real
    noise_f[-1] = noise_f[-1].real
    return np.fft.irfft(noise_f, n=n_samples).astype(np.float64)


def _add_spectral_lines(psd_vals, psd_freqs, lines_config):
    """
    Add narrow, high-power Lorentzian bumps to an analytic PSD curve at
    fixed frequencies, simulating mains hum harmonics and violin-mode-like
    resonances. Used consistently for BOTH noise generation and whatever
    gets saved as the "known" PSD -- so this is realistic contamination
    the agent has to contend with via its own PSD-division math, not an
    unfair mismatch between what's saved and what's actually in the noise.
    """
    psd_out = psd_vals.copy()
    width = lines_config["line_width_hz"]
    for f0, power_factor in lines_config["lines"]:
        # Lorentzian bump centered at f0
        lorentzian = 1.0 / (1.0 + ((psd_freqs - f0) / width) ** 2)
        psd_out = psd_out * (1.0 + (power_factor - 1.0) * lorentzian)
    return psd_out
 
 
def _apply_calibration_distortion(sig_freq_domain, freqs, calib_config, rng, severity_mult=1.0):
    """
    ...
    severity_mult scales the amplitude/phase error bounds -- used to make
    harder tiers have proportionally worse (unmodeled) calibration.
    """
    n_nodes = calib_config["n_spline_nodes"]
    amp_lo, amp_hi = calib_config["amplitude_frac_range"]
    phase_lo, phase_hi = calib_config["phase_deg_range"]
    amp_hi = amp_hi * severity_mult
    phase_hi = phase_hi * severity_mult
 
    f_min, f_max = freqs.min(), freqs.max()
    node_freqs = np.linspace(f_min, f_max, n_nodes)
    node_amp_errs = rng.uniform(-amp_hi, amp_hi, n_nodes)
    node_phase_errs_deg = rng.uniform(-phase_hi, phase_hi, n_nodes)
 
    amp_err = np.interp(freqs, node_freqs, node_amp_errs)
    phase_err_rad = np.deg2rad(np.interp(freqs, node_freqs, node_phase_errs_deg))
 
    distortion = (1.0 + amp_err) * np.exp(1j * phase_err_rad)
    return sig_freq_domain * distortion
 
 
def _inject_glitch(strain_arr, sample_rate, glitch_config, rng, segment_duration):
    """
    Inject a short, non-Gaussian sine-Gaussian burst into a time-domain
    strain array, simulating a real detector glitch. Independent of the
    astrophysical signal placement -- glitches are local, non-astrophysical
    transients that can occur anywhere in the segment.
 
    Returns (modified_strain, glitch_time_s, glitch_freq_hz, glitch_snr_like)
    for recording in ground_truth.json (hidden from the agent).
    """
    dt = 1.0 / sample_rate
    n = len(strain_arr)
    t = np.arange(n) * dt
 
    glitch_time = rng.uniform(1.0, segment_duration - 1.0)
    glitch_freq = rng.uniform(*glitch_config["freq_range_hz"])
    glitch_tau = rng.uniform(*glitch_config["duration_range_s"])
    glitch_snr_like = rng.uniform(*glitch_config["snr_range"])
 
    envelope = np.exp(-((t - glitch_time) ** 2) / (2 * glitch_tau ** 2))
    burst = envelope * np.cos(2 * np.pi * glitch_freq * (t - glitch_time))
 
    # Scale burst amplitude relative to the strain array's own noise floor
    # so glitch_snr_like is roughly interpretable across different noise levels
    noise_std = np.std(strain_arr)
    burst = burst * (glitch_snr_like * noise_std / (np.std(burst) + 1e-30))
 
    return strain_arr + burst, round(glitch_time, 4), round(glitch_freq, 2), round(glitch_snr_like, 3)



def _sample_precessing_spins(rng, precession_cfg):
    """
    Sample a physically isotropic precessing spin/orientation
    configuration (theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2).
    Tilt and viewing angle are drawn uniform-in-cosine (isotropic in 3D),
    NOT uniform in the angle itself -- this is the standard convention
    and matters physically (uniform-in-angle over-weights angles near 0
    and pi).
    """
    import math
    a_1 = rng.uniform(*precession_cfg["a_magnitude_range"])
    a_2 = rng.uniform(*precession_cfg["a_magnitude_range"])
    tilt_1 = math.acos(rng.uniform(-1.0, 1.0))
    tilt_2 = math.acos(rng.uniform(-1.0, 1.0))
    phi_12 = rng.uniform(*precession_cfg["phi_12_range"])
    phi_jl = rng.uniform(*precession_cfg["phi_jl_range"])
    theta_jn = math.acos(rng.uniform(-1.0, 1.0))
    return {
        "a_1": a_1, "a_2": a_2, "tilt_1": tilt_1, "tilt_2": tilt_2,
        "phi_12": phi_12, "phi_jl": phi_jl, "theta_jn": theta_jn,
    }
 
 
def _precessing_spins_to_cartesian(spin_params, mass1, mass2, f_lower):
    """
    Convert (theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2) to
    (iota, spin1x/y/z, spin2x/y/z) using bilby's own conversion utility --
    the SAME function used in check_waveform_residual's recovery-side
    conversion, so injection and recovery share one convention.
 
    IMPORTANT: verify bilby_to_lalsimulation_spins' exact signature with
    verify_spin_conversion.py before trusting this in a real dataset
    generation run.
    """
    from bilby.gw.conversion import bilby_to_lalsimulation_spins
    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
        theta_jn=spin_params["theta_jn"], phi_jl=spin_params["phi_jl"],
        tilt_1=spin_params["tilt_1"], tilt_2=spin_params["tilt_2"],
        phi_12=spin_params["phi_12"], a_1=spin_params["a_1"], a_2=spin_params["a_2"],
        mass_1=mass1, mass_2=mass2, reference_frequency=f_lower, phase=0.0,
    )
    return {
        "iota": float(iota),
        "spin1x": float(s1x), "spin1y": float(s1y), "spin1z": float(s1z),
        "spin2x": float(s2x), "spin2y": float(s2y), "spin2z": float(s2z),
    }
 

def generate_one_event(task_id, tier, cfg, rng, np_rng, realism_cfg=None,precession_cfg=None):
    if realism_cfg is None:
        realism_cfg = REALISM_CONFIG
    if precession_cfg is None: precession_cfg = PRECESSION_CONFIG

    target_snr  = rng.uniform(*cfg["network_snr_range"])
    ra          = rng.uniform(0, 2 * np.pi)
    dec         = rng.uniform(-np.pi / 2, np.pi / 2)
    polarisation= rng.uniform(0, np.pi)

    if COA_TIME_FRAC is None:
        coa_time_offset = rng.uniform(6.0, SEGMENT_DURATION - 2.0)
    else:
        coa_time_offset = SEGMENT_DURATION * COA_TIME_FRAC
        
    is_precessing_event = precession_cfg["enabled"]
    precessing_spin_params = None
    if is_precessing_event:
        precessing_spin_params = _sample_precessing_spins(rng, precession_cfg)
        # inclination/spin_mag from cfg are NOT used in this branch --
        # theta_jn/a_1/a_2 above replace them entirely.
    else:
        inclination = rng.uniform(*cfg["inclination_range"])
        spin_mag    = rng.uniform(*cfg["spin_magnitude_range"])
 
    injection_approximant = precession_cfg["approximant"] if is_precessing_event else APPROXIMANT

    dt        = 1.0 / SAMPLE_RATE
    n_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    flen      = n_samples // 2 + 1
    delta_f   = 1.0 / SEGMENT_DURATION

    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_L1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_vals_H1 = np.array(psd_H1)
    psd_vals_L1 = np.array(psd_L1)
    psd_freqs_arr = np.linspace(0, SAMPLE_RATE / 2, flen)
    if realism_cfg["spectral_lines"]["enabled"]:
        psd_vals_H1 = _add_spectral_lines(psd_vals_H1, psd_freqs_arr, realism_cfg["spectral_lines"])
        psd_vals_L1 = _add_spectral_lines(psd_vals_L1, psd_freqs_arr, realism_cfg["spectral_lines"])
    max_attempts = 20   # bumped up since we're now rejecting more cases
    for attempt in range(max_attempts):
        total_mass = rng.uniform(*cfg["total_mass_range"])
        mass_ratio = rng.uniform(*cfg["mass_ratio_range"])
        m1 = total_mass / (1.0 + mass_ratio)
        m2 = mass_ratio * m1
        chirp_mass = (m1 * m2)**(3.0/5.0) / (m1 + m2)**(1.0/5.0)
 
        if is_precessing_event:
            cart = _precessing_spins_to_cartesian(precessing_spin_params, m1, m2, F_LOWER)
            wf_spin_kwargs = dict(
                spin1x=cart["spin1x"], spin1y=cart["spin1y"], spin1z=cart["spin1z"],
                spin2x=cart["spin2x"], spin2y=cart["spin2y"], spin2z=cart["spin2z"],
            )
            wf_inclination = cart["iota"]  # NOT theta_jn -- iota is what the waveform generator needs
            spin1z, spin2z = cart["spin1z"], cart["spin2z"]  # for logging/backward-compat fields only
        else:
            spin1z = spin_mag * rng.choice([-1, 1])
            spin2z = spin_mag * rng.choice([-1, 1])
            wf_spin_kwargs = dict(spin1z=spin1z, spin2z=spin2z)
            wf_inclination = inclination
 

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
                approximant=injection_approximant,
                mass1=m1, mass2=m2,
                inclination=wf_inclination, coa_phase=0.0,
                delta_t=dt, f_lower=F_LOWER, distance=100.0,
                **wf_spin_kwargs,
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
                approximant=injection_approximant,
                mass1=m1, mass2=m2,
                inclination=wf_inclination, coa_phase=0.0,
                delta_t=dt, f_lower=F_LOWER, distance=distance,
                **wf_spin_kwargs,
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
        if realism_cfg["calibration_error"]["enabled"]:
            tier_scale_local = REALISM_TIER_SCALING.get(tier, {})
            calib_mult = tier_scale_local.get("calibration_mult", 1.0)
            freqs_full = np.fft.rfftfreq(n_samples, d=dt)
            sig_H1 = np.fft.irfft(
                _apply_calibration_distortion(np.fft.rfft(sig_H1), freqs_full, realism_cfg["calibration_error"], np_rng, severity_mult=calib_mult),
                n=n_samples)
            sig_L1 = np.fft.irfft(
                _apply_calibration_distortion(np.fft.rfft(sig_L1), freqs_full, realism_cfg["calibration_error"], np_rng, severity_mult=calib_mult),
                n=n_samples)
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
    glitch_time = glitch_freq = glitch_snr = glitch_detector = None
    tier_scale = REALISM_TIER_SCALING.get(tier, {})
    glitch_prob = realism_cfg["glitches"]["probability"] * tier_scale.get("glitch_probability_mult", 1.0)
    if realism_cfg["glitches"]["enabled"] and rng.random() < glitch_prob:
        which = realism_cfg["glitches"]["detector"]
        if which == "random":
            which = rng.choice(["H1", "L1"])
        if which == "H1":
            strain_H1, glitch_time, glitch_freq, glitch_snr = _inject_glitch(
                strain_H1, SAMPLE_RATE, realism_cfg["glitches"], np_rng, SEGMENT_DURATION)
        else:
            strain_L1, glitch_time, glitch_freq, glitch_snr = _inject_glitch(
                strain_L1, SAMPLE_RATE, realism_cfg["glitches"], np_rng, SEGMENT_DURATION)
        glitch_detector = which
    peak_freq_hz = min(isco_freq, SAMPLE_RATE / 2)

    difficulty_lo, difficulty_hi = cfg["difficulty_score_range"]
    difficulty_score = rng.randint(difficulty_lo, difficulty_hi)

    true_params = TrueParams(
        task_id=task_id, tier=tier, difficulty_score=difficulty_score,
        mass1=round(m1, 4), mass2=round(m2, 4),
        chirp_mass=round(chirp_mass, 4), mass_ratio=round(mass_ratio, 4),
        spin1z=round(spin1z, 4), spin2z=round(spin2z, 4),
        distance=round(distance, 2), inclination=round(wf_inclination, 4),
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
        has_glitch=glitch_time is not None,
        glitch_detector=glitch_detector,
        glitch_time_s=glitch_time,
        glitch_freq_hz=glitch_freq,
        glitch_snr_like=glitch_snr,
        spectral_lines_present=realism_cfg["spectral_lines"]["enabled"],
        calibration_error_present=realism_cfg["calibration_error"]["enabled"],
        is_precessing=is_precessing_event,
        injection_approximant=injection_approximant,
        a_1_magnitude=precessing_spin_params["a_1"] if is_precessing_event else None,
        a_2_magnitude=precessing_spin_params["a_2"] if is_precessing_event else None,
        tilt_1=precessing_spin_params["tilt_1"] if is_precessing_event else 0.0,
        tilt_2=precessing_spin_params["tilt_2"] if is_precessing_event else 0.0,
        phi_12=precessing_spin_params["phi_12"] if is_precessing_event else 0.0,
        phi_jl=precessing_spin_params["phi_jl"] if is_precessing_event else 0.0,
        theta_jn_true=precessing_spin_params["theta_jn"] if is_precessing_event else wf_inclination,
        spin1x=cart["spin1x"] if is_precessing_event else 0.0,
        spin1y=cart["spin1y"] if is_precessing_event else 0.0,
        spin2x=cart["spin2x"] if is_precessing_event else 0.0,
        spin2y=cart["spin2y"] if is_precessing_event else 0.0,
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
    detectors=["H1", "L1"], approximant_hint=injection_approximant,
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
    parser.add_argument("--enable-spin", action="store_true")
    parser.add_argument("--enable-inclination", action="store_true")
    parser.add_argument("--enable-lines", action="store_true")
    parser.add_argument("--enable-calibration-error", action="store_true")
    parser.add_argument("--enable-glitches", action="store_true")
    parser.add_argument("--enable-precession", action="store_true")
    args = parser.parse_args()

    if not PYCBC_AVAILABLE:
        print("ERROR: pycbc not installed.")
        return

    global APPROXIMANT
    APPROXIMANT = args.approximant

    import copy
    realism_cfg = copy.deepcopy(REALISM_CONFIG)
    realism_cfg["spectral_lines"]["enabled"] = args.enable_lines
    realism_cfg["calibration_error"]["enabled"] = args.enable_calibration_error
    realism_cfg["glitches"]["enabled"] = args.enable_glitches

    precession_cfg = copy.deepcopy(PRECESSION_CONFIG)
    precession_cfg["enabled"] = args.enable_precession

    # NEW: the actual saved-to-disk approximant name must reflect
    # precession injection auto-switching, not just the --approximant flag
    save_approximant = precession_cfg["approximant"] if precession_cfg["enabled"] else APPROXIMANT
    print(f"Approximant: {save_approximant}")
    print(f"Output:      {args.outdir}/{save_approximant}/")
    print(f"Note: times.npy will NOT be saved — sample_rate in task.json is sufficient.")

    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.outdir, save_approximant), exist_ok=True)

    task_counter = 0
    for tier, cfg in DIFFICULTY_CONFIG.items():
        print(f"\n--- {cfg['n_tasks']} {tier.upper()} tasks ---")
        for i in range(cfg["n_tasks"]):
            task_id = f"{task_counter:03d}"
            try:
                true_params, task_meta, arrays = generate_one_event(
                    task_id=task_id, tier=tier, cfg=cfg, rng=rng, np_rng=np_rng,
                    realism_cfg=realism_cfg, precession_cfg=precession_cfg,
                )
                save_task(args.outdir, save_approximant, true_params, task_meta, arrays)
                task_counter += 1
            except Exception as e:
                print(f"  ERROR on {task_id}: {e}")

    build_index(args.outdir, save_approximant)
    print(f"\nDone. {task_counter} tasks in {args.outdir}/{save_approximant}/")


if __name__ == "__main__":
    main()