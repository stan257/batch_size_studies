import numpy as np
import pytest

from batch_size_studies.data_loading import load_datasets, load_mnist1m_dataset


def test_load_mnist1m_dataset_adds_channel_and_scales(tmp_path):
    data_dir = tmp_path
    mnist1m_dir = data_dir / "mnist1m"
    mnist1m_dir.mkdir(parents=True)
    path = mnist1m_dir / "mnist1m.npz"

    X_train = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    y_train = np.array([0, 1, 2, 3], dtype=np.int32)
    X_test = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    y_test = np.array([4, 5], dtype=np.int32)
    np.savez(path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    (train_images, train_labels), (test_images, test_labels) = load_mnist1m_dataset(data_dir=str(data_dir))

    assert train_images.shape == (4, 4, 4, 1)
    assert test_images.shape == (2, 4, 4, 1)
    assert train_images.dtype == np.float32
    assert np.allclose(train_images[..., 0], X_train / 255.0)
    assert np.array_equal(train_labels, y_train)
    assert np.array_equal(test_labels, y_test)


def test_load_mnist1m_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mnist1m_dataset(data_dir=str(tmp_path))


def test_load_datasets_uses_tfds_builder(monkeypatch):
    class DummyBuilder:
        def __init__(self):
            self._train = {
                "image": np.ones((2, 28, 28, 1), dtype=np.uint8),
                "label": np.array([1, 2], dtype=np.int64),
            }
            self._test = {
                "image": np.zeros((1, 28, 28, 1), dtype=np.uint8),
                "label": np.array([3], dtype=np.int64),
            }

        def download_and_prepare(self):
            return None

        def as_dataset(self, split, batch_size):
            assert batch_size == -1
            return self._train if split == "train" else self._test

    monkeypatch.setattr("batch_size_studies.data_loading.tfds.builder", lambda _: DummyBuilder())
    monkeypatch.setattr("batch_size_studies.data_loading.tfds.as_numpy", lambda ds: ds)

    (train_images, train_labels), (test_images, test_labels) = load_datasets()

    assert train_images.shape == (2, 28, 28, 1)
    assert train_images.dtype == np.float32
    assert np.allclose(train_images, 1.0 / 255.0)
    assert np.array_equal(train_labels, np.array([1, 2], dtype=np.int32))
    assert test_images.shape == (1, 28, 28, 1)
    assert np.array_equal(test_labels, np.array([3], dtype=np.int32))
