import numpy as np
import pytest


@pytest.fixture
def fake_mnist1m_data_dir(tmp_path):
    """
    Creates a tiny, deterministic fake mnist1m.npz file for testing and
    returns the path to the parent data directory.
    """
    data_dir = tmp_path / "data"
    mnist_dir = data_dir / "mnist1m"
    mnist_dir.mkdir(parents=True)
    filepath = mnist_dir / "mnist1m.npz"

    # Create tiny random data. The loader expects uint8 and does the division.
    rng = np.random.default_rng(42)
    X_train = (rng.random((256, 28, 28), dtype=np.float32) * 255).astype(np.uint8)
    y_train = rng.integers(0, 10, 256, dtype=np.int32)
    X_test = (rng.random((128, 28, 28), dtype=np.float32) * 255).astype(np.uint8)
    y_test = rng.integers(0, 10, 128, dtype=np.int32)

    np.savez(filepath, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return str(data_dir)
