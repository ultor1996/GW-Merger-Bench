# GW_merger_bench/config.py

# ── Data generation constants ────────────────────────────────────────
SAMPLE_RATE      = 2048       # Hz
SEGMENT_DURATION = 16         # seconds
F_LOWER          = 20.0       # Hz — lower frequency cutoff
APPROXIMANT      = "IMRPhenomD"  # default waveform approximant

# ── Evaluation thresholds ────────────────────────────────────────────
CHIRP_MASS_TOL_FRAC = 0.05
MASS_RATIO_TOL_ABS  = 0.15
SNR_TOL_FRAC        = 0.20
OVERLAP_THRESHOLD   = 0.80
NS_MAX_MASS         = 3.0