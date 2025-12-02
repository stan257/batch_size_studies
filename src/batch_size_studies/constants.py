"""
Shared configuration constants for batch-size studies.

Keeping these values in one place avoids "magic numbers" spread across
the codebase.
"""

MNIST_EVAL_BATCH_SIZE = 512
MNIST_DEFAULT_MAX_EVAL_SAMPLES = 16_384  # 2^14
# Offset between the training PRNG stream and the evaluation subsampling stream.
# Chosen to keep the eval subset disjoint from the shuffle keys used in training.
MNIST_EVAL_SEED_OFFSET = 17

SYNTH_EVAL_MAX_SAMPLES = 10_000
# Synthetic experiments reserve two offsets: one for deterministic evaluation data,
# and another for the optional subsampling pass. Offsets are far from zero so they
# never collide with the seeds used for training batches.
SYNTH_EVAL_DATA_SEED_OFFSET = 257
SYNTH_EVAL_SUBSET_SEED_OFFSET = 259

# Runner-level offset for test-set subsampling (used outside MNISTTrialRunner).
EVAL_SUBSAMPLE_SEED_OFFSET = 1
