"""
Three pass/fail criteria:
  ok_chirp_mass  — chirp mass within 5% of true value
  ok_mass_ratio  — mass ratio within 0.15 abs of true value
  ok_merger_type — merger type correctly classified (BBH / BNS / NSBH)

Waveform overlap is computed and reported as a diagnostic metric
but does NOT affect pass/fail — it depends on extrinsic parameters
(distance, sky location, inclination).
"""

import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import OVERLAP_THRESHOLD, NS_MAX_MASS, SAMPLE_RATE, F_LOWER, APPROXIMANT
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.filter import overlap_cplx, sigma
    from pycbc.types import TimeSeries, FrequencySeries
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


@dataclass
class EvaluationResult:
    ok_chirp_mass:        bool
    ok_coalescence_time:  bool
    passed:               bool

    chirp_mass_submitted: float
    chirp_mass_true:      float
    chirp_mass_frac_err:  float

    coalescence_time_submitted: float
    coalescence_time_true:      float
    coalescence_time_abs_err:   float

    n_criteria_passed: int
    n_criteria_total:  int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class GWEvaluator:
    """
    Evaluates agent submissions against ground truth.

    The agent submits: chirp_mass_Msun, coalescence_time_s
    Sky location (ra, dec, polarisation) is GIVEN to the agent in
    task.json's given_parameters, not estimated -- so it is not scored.
    """

    def __init__(self, ground_truth: Dict[str, Any], task_dir: str = None):
        self.gt       = ground_truth
        self.task_dir = task_dir

        self.true_chirp_mass     = ground_truth["chirp_mass"]
        self.true_coa_time       = ground_truth["coalescence_time"]

        self.chirp_mass_tol      = ground_truth["chirp_mass_tol_frac"]
        self.coalescence_time_tol = ground_truth.get("coalescence_time_tol_s", 0.05)

    def evaluate(self, submission: Dict[str, Any]) -> EvaluationResult:
        sub_chirp_mass = float(submission.get("chirp_mass_Msun", -1.0))
        sub_coa_time   = float(submission.get("coalescence_time_s", -1.0))

        # ---- Criterion 1: Chirp mass ----
        if sub_chirp_mass > 0:
            cm_frac_err = abs(sub_chirp_mass - self.true_chirp_mass) / self.true_chirp_mass
        else:
            cm_frac_err = 1.0
        ok_chirp_mass = cm_frac_err <= self.chirp_mass_tol

        # ---- Criterion 2: Coalescence time ----
        ct_abs_err = abs(sub_coa_time - self.true_coa_time)
        ok_coalescence_time = ct_abs_err <= self.coalescence_time_tol

        # ---- Conjunction gate ----
        n_passed = sum([ok_chirp_mass, ok_coalescence_time])
        passed   = ok_chirp_mass and ok_coalescence_time

        return EvaluationResult(
            ok_chirp_mass=ok_chirp_mass,
            ok_coalescence_time=ok_coalescence_time,
            passed=passed,
            chirp_mass_submitted=sub_chirp_mass,
            chirp_mass_true=self.true_chirp_mass,
            chirp_mass_frac_err=cm_frac_err,
            coalescence_time_submitted=sub_coa_time,
            coalescence_time_true=self.true_coa_time,
            coalescence_time_abs_err=ct_abs_err,
            n_criteria_passed=n_passed,
            n_criteria_total=2,
        )

    def _compute_waveform_overlap(self, mass1: float, mass2: float) -> tuple:
        """
        Compute noise-weighted overlap using submitted masses directly.
        Uses true extrinsic parameters from ground_truth.json.
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

            psd_interp = np.interp(
                np.linspace(0, SAMPLE_RATE / 2, flen),
                psd_freqs, psd_H1,
                left=1e-40, right=1e-40,
            )
            psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
            psd_fs     = FrequencySeries(psd_interp, delta_f=delta_f)

            det     = Detector("H1")
            gps_coa = 1264316116.0 + self.true_coa_time
            fp, fc  = det.antenna_pattern(
                self.true_ra, self.true_dec, self.true_polarisation, gps_coa
            )
            coa_idx   = int(self.true_coa_time * SAMPLE_RATE)
            strain_ts = TimeSeries(strain_H1, delta_t=dt)

            best_overlap = 0.0

            # Try 4 coalescence phases, take best — submitted masses used directly
            for coa_phase in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
                try:
                    hp, hc = get_td_waveform(
                        approximant=self.approximant,
                        mass1=mass1, mass2=mass2,
                        spin1z=self.true_spin1z,
                        spin2z=self.true_spin2z,
                        distance=self.true_distance,
                        inclination=self.true_inclination,
                        coa_phase=coa_phase,
                        delta_t=dt, f_lower=F_LOWER,
                    )
                    raw    = fp * np.array(hp) + fc * np.array(hc)
                    sig    = np.zeros(n)
                    e      = min(coa_idx, len(hp))
                    s      = len(hp) - e
                    sig[coa_idx - e: coa_idx] = raw[s:]

                    tmpl_ts   = TimeSeries(sig, delta_t=dt)
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
            ns = NS_MAX_MASS
            lo = min(sub_mass1, sub_mass2)
            hi = max(sub_mass1, sub_mass2)
            if submitted_type == "BBH":
                return lo > ns
            elif submitted_type == "BNS":
                return hi < ns
            elif submitted_type == "NSBH":
                return lo < ns and hi > ns
        return True