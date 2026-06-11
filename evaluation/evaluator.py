"""
GW Merger Bench — Evaluator
Conjunction gate with four criteria.

Four criteria:
  ok_waveform_match — noise-weighted overlap >= 0.90
                      Uses submitted masses but TRUE extrinsic params
                      (distance, sky, inclination, spins) since the agent
                      cannot recover those from single-detector matched filtering.
  ok_chirp_mass     — chirp mass within 5% of true value
  ok_mass_ratio     — mass ratio within 0.15 abs of true value
  ok_merger_type    — merger type correctly classified (BBH / BNS / NSBH)

Statistical vs physical gap:
  ok_waveform_match can pass while ok_chirp_mass fails.
  This is the core "good statistics != good physics" signal.
"""

import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.filter import overlap_cplx, sigma
    from pycbc.types import TimeSeries, FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False

SAMPLE_RATE       = 2048
F_LOWER           = 20.0
APPROXIMANT       = "IMRPhenomD"
OVERLAP_THRESHOLD = 0.90


@dataclass
class EvaluationResult:
    ok_waveform_match: bool
    ok_chirp_mass:     bool
    ok_mass_ratio:     bool
    ok_merger_type:    bool
    passed:            bool

    waveform_overlap:           float
    waveform_overlap_threshold: float

    chirp_mass_submitted: float
    chirp_mass_true:      float
    chirp_mass_frac_err:  float

    mass_ratio_submitted: float
    mass_ratio_true:      float
    mass_ratio_abs_err:   float

    merger_type_submitted: str
    merger_type_true:      str

    n_criteria_passed: int
    n_criteria_total:  int = 4

    anchor_chirp_mass_from_freq_evo: float = 0.0
    anchor_peak_freq_hz:             float = 0.0
    stat_pass_phys_fail:             bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GWEvaluator:
    """
    Evaluates agent submissions against ground truth.

    The agent only submits recoverable parameters:
      chirp_mass_Msun, mass1_Msun, mass2_Msun, mass_ratio,
      network_snr, merger_type

    For ok_waveform_match the evaluator uses submitted masses but
    TRUE extrinsic parameters (distance, sky location, inclination,
    spins) since these cannot be recovered from single-detector
    matched filtering.
    """

    def __init__(self, ground_truth: Dict[str, Any], task_dir: str = None):
        self.gt       = ground_truth
        self.task_dir = task_dir

        self.true_chirp_mass  = ground_truth["chirp_mass"]
        self.true_mass_ratio  = ground_truth["mass_ratio"]
        self.true_merger_type = ground_truth["merger_type"]
        self.true_mass1       = ground_truth["mass1"]
        self.true_mass2       = ground_truth["mass2"]

        # True extrinsic params — used by evaluator for waveform overlap
        self.true_coa_time    = ground_truth["coalescence_time"]
        self.true_polarisation= ground_truth["polarisation"]
        self.true_distance    = ground_truth["distance"]
        self.true_inclination = ground_truth["inclination"]
        self.true_ra          = ground_truth["ra"]
        self.true_dec         = ground_truth["dec"]
        self.true_spin1z      = ground_truth["spin1z"]
        self.true_spin2z      = ground_truth["spin2z"]

        self.approximant      = ground_truth.get("approximant", APPROXIMANT)
        self.anchor_chirp_mass= ground_truth["chirp_mass_from_freq_evo"]
        self.anchor_peak_freq = ground_truth["peak_frequency_hz"]
        self.chirp_mass_tol   = ground_truth["chirp_mass_tol_frac"]
        self.mass_ratio_tol   = ground_truth["mass_ratio_tol_abs"]

        self._strain_H1 = None
        self._psd_H1    = None
        self._psd_freqs = None

    def evaluate(self, submission: Dict[str, Any]) -> EvaluationResult:
        sub_chirp_mass  = float(submission.get("chirp_mass_Msun", -1.0))
        sub_mass_ratio  = float(submission.get("mass_ratio", -1.0))
        sub_mass1       = float(submission.get("mass1_Msun", 0.0))
        sub_mass2       = float(submission.get("mass2_Msun", 0.0))
        sub_merger_type = str(submission.get("merger_type", "")).strip().upper()

        # Derive masses from chirp mass if not provided
        if sub_mass1 <= 0 or sub_mass2 <= 0:
            sub_mass1, sub_mass2 = self._masses_from_chirp(sub_chirp_mass, sub_mass_ratio)

        # ---- Criterion 1: Waveform match ----
        # Uses submitted masses but TRUE extrinsic params
        overlap, ok_waveform = self._compute_waveform_overlap(
            mass1=sub_mass1,
            mass2=sub_mass2,
        )

        # ---- Criterion 2: Chirp mass ----
        if sub_chirp_mass > 0:
            cm_frac_err = abs(sub_chirp_mass - self.true_chirp_mass) / self.true_chirp_mass
        else:
            cm_frac_err = 1.0
        ok_chirp_mass = cm_frac_err <= self.chirp_mass_tol

        # ---- Criterion 3: Mass ratio ----
        sub_mass_ratio_clipped = max(min(sub_mass_ratio, 1.0), 0.0)
        mr_abs_err  = abs(sub_mass_ratio_clipped - self.true_mass_ratio)
        ok_mass_ratio = mr_abs_err <= self.mass_ratio_tol

        # ---- Criterion 4: Merger type ----
        ok_merger_type = self._check_merger_type(sub_merger_type, sub_mass1, sub_mass2)

        # ---- Conjunction gate ----
        n_passed = sum([ok_waveform, ok_chirp_mass, ok_mass_ratio, ok_merger_type])
        passed   = ok_waveform and ok_chirp_mass and ok_mass_ratio and ok_merger_type

        return EvaluationResult(
            ok_waveform_match=ok_waveform,
            ok_chirp_mass=ok_chirp_mass,
            ok_mass_ratio=ok_mass_ratio,
            ok_merger_type=ok_merger_type,
            passed=passed,
            waveform_overlap=round(float(overlap), 4),
            waveform_overlap_threshold=OVERLAP_THRESHOLD,
            chirp_mass_submitted=sub_chirp_mass,
            chirp_mass_true=self.true_chirp_mass,
            chirp_mass_frac_err=cm_frac_err,
            mass_ratio_submitted=sub_mass_ratio,
            mass_ratio_true=self.true_mass_ratio,
            mass_ratio_abs_err=mr_abs_err,
            merger_type_submitted=sub_merger_type,
            merger_type_true=self.true_merger_type,
            n_criteria_passed=n_passed,
            n_criteria_total=4,
            anchor_chirp_mass_from_freq_evo=self.anchor_chirp_mass,
            anchor_peak_freq_hz=self.anchor_peak_freq,
            stat_pass_phys_fail=ok_waveform and not ok_chirp_mass,
        )

    def _compute_waveform_overlap(self, mass1: float, mass2: float) -> tuple:
        """
        Compute noise-weighted overlap using submitted masses and
        TRUE extrinsic parameters. Returns (overlap, passed).
        """
        if not PYCBC_AVAILABLE or self.task_dir is None:
            return 0.0, False

        try:
            strain_H1, psd_H1, psd_freqs = self._load_data()

            dt      = 1.0 / SAMPLE_RATE
            n       = len(strain_H1)
            delta_f = 1.0 / (n * dt)
            flen    = n // 2 + 1

            mass1 = max(float(mass1), 1.0)
            mass2 = max(float(mass2), 1.0)
            if mass1 < mass2:
                mass1, mass2 = mass2, mass1

            # Build PSD FrequencySeries
            psd_interp = np.interp(
                np.linspace(0, SAMPLE_RATE / 2, flen),
                psd_freqs, psd_H1,
                left=1e-40, right=1e-40,
            )
            psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
            psd_fs     = FrequencySeries(psd_interp, delta_f=delta_f)

            # Detector projection using TRUE extrinsic params
            det     = Detector("H1")
            gps_coa = 1264316116.0 + self.true_coa_time
            fp, fc  = det.antenna_pattern(
                self.true_ra, self.true_dec, self.true_polarisation, gps_coa
            )

            coa_idx   = int(self.true_coa_time * SAMPLE_RATE)
            strain_ts = TimeSeries(strain_H1, delta_t=dt)

            # Try 4 coalescence phases — take best overlap
            best_overlap = 0.0
            for coa_phase in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                try:
                    hp2, hc2 = get_td_waveform(
                        approximant=self.approximant,
                        mass1=mass1, mass2=mass2,
                        spin1z=self.true_spin1z,
                        spin2z=self.true_spin2z,
                        distance=self.true_distance,
                        inclination=self.true_inclination,
                        coa_phase=coa_phase,
                        delta_t=dt, f_lower=F_LOWER,
                    )
                    raw2  = fp * np.array(hp2) + fc * np.array(hc2)
                    sig2  = np.zeros(n)
                    e2    = min(coa_idx, len(hp2))
                    s2    = len(hp2) - e2
                    sig2[coa_idx - e2: coa_idx] = raw2[s2:]

                    tmpl_ts   = TimeSeries(sig2, delta_t=dt)
                    tmpl_norm = float(sigma(tmpl_ts, psd=psd_fs,
                                           low_frequency_cutoff=F_LOWER))
                    if tmpl_norm < 1e-30:
                        continue

                    ov = overlap_cplx(strain_ts, tmpl_ts, psd=psd_fs,
                                      low_frequency_cutoff=F_LOWER,
                                      normalized=True)
                    ov_abs = float(abs(ov))
                    if ov_abs > best_overlap:
                        best_overlap = ov_abs
                except Exception:
                    continue

            return best_overlap, best_overlap >= OVERLAP_THRESHOLD

        except Exception:
            return 0.0, False

    def _load_data(self):
        if self._strain_H1 is None:
            self._strain_H1 = np.load(os.path.join(self.task_dir, "strain_H1.npy"))
            self._psd_H1    = np.load(os.path.join(self.task_dir, "psd_H1.npy"))
            self._psd_freqs = np.load(os.path.join(self.task_dir, "psd_freqs.npy"))
        return self._strain_H1, self._psd_H1, self._psd_freqs

    def _masses_from_chirp(self, chirp_mass: float, mass_ratio: float):
        if chirp_mass <= 0 or mass_ratio <= 0:
            return 30.0, 20.0
        q     = min(max(mass_ratio, 0.01), 1.0)
        total = chirp_mass * ((1 + q) ** 1.2) / (q ** 0.6)
        m2    = total * q / (1 + q)
        m1    = total / (1 + q)
        return max(m1, m2), min(m1, m2)

    def _check_merger_type(self, submitted_type, sub_mass1, sub_mass2) -> bool:
        if submitted_type != self.true_merger_type:
            return False
        if sub_mass1 > 0 and sub_mass2 > 0:
            ns = 3.0
            lo = min(sub_mass1, sub_mass2)
            hi = max(sub_mass1, sub_mass2)
            if submitted_type == "BBH":
                return lo > ns
            elif submitted_type == "BNS":
                return hi < ns
            elif submitted_type == "NSBH":
                return lo < ns and hi > ns
        return True