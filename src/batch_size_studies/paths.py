import os


def get_project_root() -> str:
    """
    Finds the project root by assuming a fixed directory structure.
    This makes path resolution robust to the script's execution location.
    """
    # This file is in .../src/batch_size_studies/
    # The project root is two directories up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# The absolute path to the project root directory.
PROJECT_ROOT = get_project_root()

# Default directory for storing experiment results, relative to the project root.
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Default directory for storing datasets, relative to the project root.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure the directories exist so that scripts can write to them.
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
