"""
GW Merger Bench — Evaluator
Conjunction gate with four criteria mirroring Stargazer's design.

Four criteria:
  ok_waveform_match — noise-weighted overlap between reconstructed and observed strain >= 0.90
                      Evaluator generates a clean template from submitted parameters using
                      IMRPhenomD and computes the overlap with the actual strain data.
                      Agent cannot fake this — wrong parameters produce wrong waveform shape.
  ok_chirp_mass     — chirp mass within 5% of true value
  ok_mass_ratio     — mass ratio within 0.15 abs of true value
  ok_merger_type    — merger type correctly classified

Statistical vs physical gap:
  ok_waveform_match can pass (good data fit) while ok_chirp_mass fails (wrong physics).
  This is the core "good statistics != good physics" signal.
"""

import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Try to import pycbc — graceful fallback if not available
# ---------------------------------------------------------------------------
try:
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.filter import matched_filter, overlap_cplx, sigma
    from pycbc.types import TimeSeries, FrequencySeries
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    import pycbc.psd as pycbc_psd
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


SAMPLE_RATE   = 2048
F_LOWER       = 20.0
APPROXIMANT   = "IMRPhenomD"  # fallback only — evaluator reads from ground_truth.json
OVERLAP_THRESHOLD = 0.90   # minimum noise-weighted overlap to pass ok_waveform_match


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    # Per-criterion pass/fail
    ok_waveform_match: bool
    ok_chirp_mass: bool
    ok_mass_ratio: bool
    ok_merger_type: bool

    # Conjunction gate
    passed: bool

    # Waveform match score (0–1, computed by evaluator from forward model)
    waveform_overlap: float
    waveform_overlap_threshold: float

    # Chirp mass error
    chirp_mass_submitted: float
    chirp_mass_true: float
    chirp_mass_frac_err: float

    # Mass ratio error
    mass_ratio_submitted: float
    mass_ratio_true: float
    mass_ratio_abs_err: float

    # Merger type
    merger_type_submitted: str
    merger_type_true: str

    n_criteria_passed: int
    n_criteria_total: int = 4

    # Anchor values (recomputed from forward model)
    anchor_chirp_mass_from_freq_evo: float = 0.0
    anchor_peak_freq_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class GWEvaluator:
    """
    Evaluates agent submissions against ground truth.

    ok_waveform_match is computed by the evaluator itself:
      1. Generate clean template from submitted parameters using IMRPhenomD
      2. Project onto H1 detector using submitted sky location
      3. Compute noise-weighted overlap with actual strain_H1.npy
      4. Pass if overlap >= OVERLAP_THRESHOLD (default 0.90)

    This mirrors Stargazer's ok_match — the evaluator re-runs the forward
    model independently so the agent cannot self-report this metric.
    """

    def __init__(self, ground_truth: Dict[str, Any], task_dir: str = None):
        self.gt       = ground_truth
        self.task_dir = task_dir

        # Ground truth values
        self.true_chirp_mass  = ground_truth["chirp_mass"]
        self.true_mass_ratio  = ground_truth["mass_ratio"]
        self.true_merger_type = ground_truth["merger_type"]
        self.true_mass1       = ground_truth["mass1"]
        self.true_mass2       = ground_truth["mass2"]
        self.true_coa_time    = ground_truth["coalescence_time"]
        self.true_polarisation= ground_truth["polarisation"]

        # Waveform model — must match what was used to generate the data
        self.approximant = ground_truth.get("approximant", APPROXIMANT)

        # Anchor values
        self.anchor_chirp_mass = ground_truth["chirp_mass_from_freq_evo"]
        self.anchor_peak_freq  = ground_truth["peak_frequency_hz"]

        # Tolerances
        self.chirp_mass_tol = ground_truth["chirp_mass_tol_frac"]
        self.mass_ratio_tol = ground_truth["mass_ratio_tol_abs"]

        # Cache strain and PSD once loaded
        self._strain_H1 = None
        self._psd_H1    = None
        self._psd_freqs = None

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def evaluate(self, submission: Dict[str, Any]) -> EvaluationResult:

        sub_chirp_mass   = float(submission.get("chirp_mass_Msun", -1.0))
        sub_mass_ratio   = float(submission.get("mass_ratio", -1.0))
        sub_mass1        = float(submission.get("mass1_Msun", 0.0))
        sub_mass2        = float(submission.get("mass2_Msun", 0.0))
        sub_distance     = float(submission.get("distance_Mpc", 500.0))
        sub_inclination  = float(submission.get("inclination_rad", 0.4))
        sub_ra           = float(submission.get("ra_rad", 0.0))
        sub_dec          = float(submission.get("dec_rad", 0.0))
        sub_spin1z       = float(submission.get("spin1z", 0.0))
        sub_spin2z       = float(submission.get("spin2z", 0.0))
        sub_merger_type  = str(submission.get("merger_type", "")).strip().upper()

        # ---- Criterion 1: Waveform match (forward model recompute) ----
        overlap, ok_waveform = self._compute_waveform_overlap(
            mass1=sub_mass1 if sub_mass1 > 0 else self._masses_from_chirp(sub_chirp_mass, sub_mass_ratio)[0],
            mass2=sub_mass2 if sub_mass2 > 0 else self._masses_from_chirp(sub_chirp_mass, sub_mass_ratio)[1],
            spin1z=sub_spin1z,
            spin2z=sub_spin2z,
            distance=sub_distance,
            inclination=sub_inclination,
            ra=sub_ra,
            dec=sub_dec,
        )

        # ---- Criterion 2: Chirp mass ----
        if sub_chirp_mass > 0:
            cm_frac_err = abs(sub_chirp_mass - self.true_chirp_mass) / self.true_chirp_mass
        else:
            cm_frac_err = 1.0
        ok_chirp_mass = cm_frac_err <= self.chirp_mass_tol

        # ---- Criterion 3: Mass ratio ----
        sub_mass_ratio_clipped = max(min(sub_mass_ratio, 1.0), 0.0)
        mr_abs_err = abs(sub_mass_ratio_clipped - self.true_mass_ratio)
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
        )

    # ------------------------------------------------------------------
    # Waveform overlap computation
    # ------------------------------------------------------------------
    def _compute_waveform_overlap(
        self,
        mass1: float,
        mass2: float,
        spin1z: float,
        spin2z: float,
        distance: float,
        inclination: float,
        ra: float,
        dec: float,
    ) -> tuple:
        """
        Generate clean template from submitted parameters using IMRPhenomD.
        Project onto H1. Compute noise-weighted overlap with actual strain_H1.npy.
        Returns (overlap_value, passed_bool).
        Falls back to 0.0 overlap if pycbc unavailable or waveform generation fails.
        """
        if not PYCBC_AVAILABLE or self.task_dir is None:
            return 0.0, False

        try:
            # Load strain and PSD (cached after first load)
            strain_H1, psd_H1, psd_freqs = self._load_data()

            dt    = 1.0 / SAMPLE_RATE
            n     = len(strain_H1)
            delta_f = 1.0 / (n * dt)
            flen  = n // 2 + 1

            # Ensure masses are physically valid
            mass1 = max(mass1, 1.0)
            mass2 = max(mass2, 1.0)
            if mass1 < mass2:
                mass1, mass2 = mass2, mass1
            spin1z = float(np.clip(spin1z, -0.99, 0.99))
            spin2z = float(np.clip(spin2z, -0.99, 0.99))
            distance = max(distance, 1.0)

            # Generate clean template
            hp, hc = get_td_waveform(
                approximant=self.approximant,
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                distance=distance,
                inclination=inclination,
                delta_t=dt,
                f_lower=F_LOWER,
            )

            # Project onto H1 using submitted sky location
            det      = Detector("H1")
            gps_coa  = 1264316116.0 + self.true_coa_time
            fp, fc   = det.antenna_pattern(ra, dec, self.true_polarisation, gps_coa)

            # Build projected strain array aligned to segment
            sig_arr  = np.zeros(n)
            coa_idx  = int(self.true_coa_time * SAMPLE_RATE)
            end_idx  = min(coa_idx, len(hp))
            start_src= len(hp) - end_idx
            raw      = fp * np.array(hp) + fc * np.array(hc)
            sig_arr[coa_idx - end_idx: coa_idx] = raw[start_src:]

            # Build PSD FrequencySeries
            psd_interp = np.interp(
                np.linspace(0, SAMPLE_RATE / 2, flen),
                psd_freqs,
                psd_H1,
                left=1e-40, right=1e-40,
            )
            psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
            psd_fs = FrequencySeries(psd_interp, delta_f=delta_f)

            # Try multiple coalescence phases — take best overlap
            # (phase is a free parameter, conventionally marginalised)
            best_overlap = 0.0
            for coa_phase_offset in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                try:
                    hp2, hc2 = get_td_waveform(
                        approximant=self.approximant,
                        mass1=mass1,
                        mass2=mass2,
                        spin1z=spin1z,
                        spin2z=spin2z,
                        distance=distance,
                        inclination=inclination,
                        coa_phase=coa_phase_offset,
                        delta_t=dt,
                        f_lower=F_LOWER,
                    )
                    raw2   = fp * np.array(hp2) + fc * np.array(hc2)
                    sig2   = np.zeros(n)
                    e2     = min(coa_idx, len(hp2))
                    s2     = len(hp2) - e2
                    sig2[coa_idx - e2: coa_idx] = raw2[s2:]

                    tmpl_ts  = TimeSeries(sig2, delta_t=dt)
                    strain_ts = TimeSeries(strain_H1, delta_t=dt)

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

            passed = best_overlap >= OVERLAP_THRESHOLD
            return best_overlap, passed

        except Exception as e:
            # Any failure → 0 overlap, criterion fails
            return 0.0, False

    # ------------------------------------------------------------------
    # Data loading (cached)
    # ------------------------------------------------------------------
    def _load_data(self):
        if self._strain_H1 is None:
            self._strain_H1 = np.load(
                os.path.join(self.task_dir, "strain_H1.npy"))
            self._psd_H1 = np.load(
                os.path.join(self.task_dir, "psd_H1.npy"))
            self._psd_freqs = np.load(
                os.path.join(self.task_dir, "psd_freqs.npy"))
        return self._strain_H1, self._psd_H1, self._psd_freqs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _masses_from_chirp(self, chirp_mass: float, mass_ratio: float):
        """Derive m1, m2 from chirp mass and mass ratio."""
        if chirp_mass <= 0 or mass_ratio <= 0:
            return 30.0, 20.0
        q = min(max(mass_ratio, 0.01), 1.0)
        # Mc = (m1*m2)^0.6 / (m1+m2)^0.2,  q = m2/m1
        # m1 + m2 = Mc * (1+q)^1.2 / q^0.6
        total = chirp_mass * ((1 + q) ** 1.2) / (q ** 0.6)
        m2 = total * q / (1 + q)
        m1 = total / (1 + q)
        return max(m1, m2), min(m1, m2)

    def _check_merger_type(self, submitted_type, sub_mass1, sub_mass2) -> bool:
        if submitted_type != self.true_merger_type:
            return False
        if sub_mass1 > 0 and sub_mass2 > 0:
            ns  = 3.0
            lo  = min(sub_mass1, sub_mass2)
            hi  = max(sub_mass1, sub_mass2)
            if submitted_type == "BBH":
                return lo > ns
            elif submitted_type == "BNS":
                return hi < ns
            elif submitted_type == "NSBH":
                return lo < ns and hi > ns
        return True

    def anchor_check(self, anchor_name: str, claimed_value: float) -> Dict[str, Any]:
        anchors = {
            "chirp_mass":         {"true": self.true_chirp_mass,  "tol": self.chirp_mass_tol, "err_type": "fractional"},
            "chirp_mass_freq_evo":{"true": self.anchor_chirp_mass, "tol": self.chirp_mass_tol, "err_type": "fractional"},
            "mass_ratio":         {"true": self.true_mass_ratio,   "tol": self.mass_ratio_tol, "err_type": "absolute"},
            "peak_frequency_hz":  {"true": self.anchor_peak_freq,  "tol": 0.10,                "err_type": "fractional"},
        }
        if anchor_name not in anchors:
            return {"passed": None, "error": f"Unknown anchor: {anchor_name}"}
        cfg = anchors[anchor_name]
        err = (abs(claimed_value - cfg["true"]) / max(abs(cfg["true"]), 1e-10)
               if cfg["err_type"] == "fractional"
               else abs(claimed_value - cfg["true"]))
        return {
            "anchor":        anchor_name,
            "passed":        err <= cfg["tol"],
            "true_value":    cfg["true"],
            "claimed_value": claimed_value,
            "error":         round(err, 6),
            "error_type":    cfg["err_type"],
            "tolerance":     cfg["tol"],
        }