# GW_merger_bench/config.py

# ── Data generation constants ────────────────────────────────────────
SAMPLE_RATE      = 2048       # Hz
SEGMENT_DURATION = 16         # seconds
F_LOWER          = 20.0       # Hz — lower frequency cutoff
APPROXIMANT      = "IMRPhenomD"  # default waveform approximant
DISTANCE_RANGE_MPC = {
    "easy":   (100.0,  2000.0),
    "medium": (500.0,  4000.0),
    "hard":   (1000.0, 8000.0),
    "calibration": (100, 2000),
}

# ── Evaluation thresholds ────────────────────────────────────────────
CHIRP_MASS_TOL_FRAC = 0.05
MASS_RATIO_TOL_ABS  = 0.15
SNR_TOL_FRAC        = 0.20
OVERLAP_THRESHOLD   = 0.80
NS_MAX_MASS         = 3.0


# ── Signal placement ─────────────────────────────────────────────────
# Coalescence time as a fraction of segment duration.
# 0.67 = merger at 67% of the 16s segment = 10.72s
# Fixed for controlled benchmark — all tasks have merger at same time.
# Set to None to sample randomly between 2s and (SEGMENT_DURATION-2s).
COA_TIME_FRAC = None

# ── Noise ────────────────────────────────────────────────────────────
# Seed strategy for coloured Gaussian noise per task.
# "task_id" — deterministic per task_id (default, reproducible)
# "random"  — different noise every time generate_dataset.py runs
# integer   — fixed seed for all tasks (same noise for every task)
NOISE_SEED_STRATEGY = "task_id"