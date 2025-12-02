"""
Shared configuration constants for batch-size studies.

Keeping these values in one place avoids "magic numbers" spread across
the codebase.
"""

MNIST_EVAL_BATCH_SIZE = 512
MNIST_DEFAULT_MAX_EVAL_SAMPLES = 16_384  # 2^14 for the MNIST-1M test subset.
MNIST_EVAL_SEED_OFFSET = 17  # Offset from init_key for MNIST eval subsampling.

SYNTH_EVAL_MAX_SAMPLES = 10_000
SYNTH_EVAL_DATA_SEED_OFFSET = 257  # Offset for deterministic synthetic eval data.
SYNTH_EVAL_SUBSET_SEED_OFFSET = 259  # Offset for synthetic eval subsampling.

EVAL_SUBSAMPLE_SEED_OFFSET = 1  # Runner-level offset for test-set subsampling.
