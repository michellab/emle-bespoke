#!/usr/bin/env python
"""
Merge datasets from different sources into a single dataset.
"""

import argparse
import os
import pickle
from typing import Any, Dict, List


def load_dataset(file_path: str) -> Dict[Any, Any]:
    """
    Load a dataset from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to load dataset from {file_path}: {e}")


def merge_datasets(datasets: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """
    Merge multiple datasets into one.

    Args:
        datasets (List[Dict[Any, Any]]): List of datasets to merge.

    Returns:
        dict: Merged dataset.
    """
    merged = {}
    for dataset in datasets:
        for key, value in dataset.items():
            if key in merged:
                merged[key] += value
            else:
                merged[key] = value
    return merged


def save_dataset(data: Dict[Any, Any], output_path: str) -> None:
    """
    Save a dataset to a pickle file.

    Args:
        data (dict): Dataset to save.
        output_path (str): Path to the output pickle file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Merge datasets from different sources into a single dataset."
    )

    # Command-line arguments
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the source datasets.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output_ref_data.pkl",
        help="Path to the output dataset.",
    )

    args = parser.parse_args()

    # Load the datasets
    try:
        datasets = [load_dataset(file) for file in args.files]
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Merge the datasets
    merged_dataset = merge_datasets(datasets)

    # Save the merged dataset
    try:
        save_dataset(merged_dataset, args.output)
        print(f"Merged dataset saved to {args.output}")
    except Exception as e:
        print(f"Error saving merged dataset: {e}")


if __name__ == "__main__":
    main()
