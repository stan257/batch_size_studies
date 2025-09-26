import os

import numpy as np
import tensorflow_datasets as tfds

from .paths import DATA_DIR


def load_datasets():
    """Loads the standard MNIST dataset using tensorflow_datasets."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    # Return raw numpy arrays, conversion to JAX arrays will happen in the runner.
    train_images = train_ds["image"].astype(np.float32) / 255.0
    train_labels = train_ds["label"].astype(np.int32)
    test_images = test_ds["image"].astype(np.float32) / 255.0
    test_labels = test_ds["label"].astype(np.int32)

    return (train_images, train_labels), (test_images, test_labels)


def load_mnist1m_dataset(data_dir=DATA_DIR):
    """
    Loads the pre-processed MNIST-1M dataset from a local .npz file.
    """
    dataset_path = os.path.join(data_dir, "mnist1m", "mnist1m.npz")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"MNIST-1M dataset not found at '{dataset_path}'. "
            f"Please run `python scripts/process_mnist1m.py` script first."
        )

    with np.load(dataset_path) as data:
        X_train = data["X_train"].astype(np.float32) / 255.0
        y_train = data["y_train"].astype(np.int32)
        X_test = data["X_test"].astype(np.float32) / 255.0
        y_test = data["y_test"].astype(np.int32)

    # Add a channel dimension for compatibility with the training loop,
    # which expects a 4D tensor.
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if X_test.ndim == 3:
        X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)
