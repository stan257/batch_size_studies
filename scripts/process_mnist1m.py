import argparse
import os
import zipfile

import numpy as np

# --- Check for dependencies and provide helpful error messages ---
try:
    from PIL import Image
except ImportError:
    print("Error: The 'Pillow' library is required. Please install it by running: pip install Pillow")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: The 'tqdm' library is required. Please install it by running: pip install tqdm")
    exit(1)

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Error: The 'scikit-learn' library is required. Please install it by running: pip install scikit-learn")
    exit(1)


def process_mnist1m(raw_data_dir, output_dir, test_size=50000, seed=42):
    """
    Processes the raw MNIST-1M zip files into a single compressed NPZ file.

    This function performs a one-time conversion:
    1. Unzips images from 10 class-specific zip files.
    2. Loads 1M images and assigns labels.
    3. Performs a standard, reproducible train/test split.
    4. Saves the data as 'mnist1m.npz' in the specified output directory.

    Args:
        raw_data_dir (str): The directory containing the 10 raw zip folders (e.g., 'data/mnist1m/raw').
        output_dir (str): The directory to save the final 'mnist1m.npz' file (e.g., 'data/mnist1m').
        test_size (int): The number of samples to allocate for the test set.
        seed (int): The random seed for the train/test split to ensure reproducibility.
    """
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found at '{raw_data_dir}'")
        print("Please place the 10 class folders (0-9) inside it.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mnist1m.npz")

    if os.path.exists(output_path):
        print(f"Processed file already exists at '{output_path}'. Skipping.")
        return

    all_images = []
    all_labels = []

    print("--- Starting MNIST-1M Pre-processing ---")
    # Iterate through class folders 0 to 9
    for class_label in range(10):
        zip_filename = f"class_{class_label}_images.zip"
        zip_filepath = os.path.join(raw_data_dir, zip_filename)

        if not os.path.exists(zip_filepath):
            print(f"Warning: Zip file not found for class {class_label} at '{zip_filepath}'. Skipping.")
            continue

        print(f"Processing images for class {class_label}...")
        with zipfile.ZipFile(zip_filepath, "r") as zf:
            image_files = [f for f in zf.namelist() if f.endswith((".png", ".jpg", ".jpeg"))]

            for filename in tqdm(image_files, desc=f"Class {class_label}", leave=False):
                with zf.open(filename) as img_file:
                    with Image.open(img_file) as img:
                        img = img.convert("L").resize((28, 28))
                        all_images.append(np.array(img))
                        all_labels.append(class_label)

    if not all_images:
        print("No images were processed. Aborting.")
        return

    X = np.array(all_images, dtype=np.uint8)
    y = np.array(all_labels, dtype=np.uint8)

    print(f"\nTotal images processed: {len(X)}")
    print("Performing train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size:  {len(X_test)}")

    print(f"Saving processed data to '{output_path}'...")
    # Use np.savez for faster loading, at the cost of a larger file size.
    np.savez(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("--- Pre-processing complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process the MNIST-1M dataset from raw zip files.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/mnist1m/raw",
        help="Directory containing the 10 raw class folders with zip files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mnist1m",
        help="Directory to save the final 'mnist1m.npz' file.",
    )
    args = parser.parse_args()

    process_mnist1m(args.raw_dir, args.output_dir)
