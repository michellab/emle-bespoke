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

    Parameters
    ----------
    file_path : str
        Path to the pickle file.

    Returns
    -------
    dict
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be unpickled.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(f"Failed to load dataset from {file_path}: {e}")


def merge_datasets(
    datasets: List[Dict[Any, Any]],
    start: int,
    end: int,
    stride: int,
    write_complementary: bool = False,
) -> Dict[str, Any]:
    """
    Merge multiple datasets into one.

    Parameters
    ----------
    datasets: List[Dict[Any, Any]]
        List of datasets to merge.
    start: int
        Index of the first sample to include from each dataset.
    end: int
        Index of the last sample to include from each dataset.
    stride: int
        Stride for selecting samples from each dataset.
    write_complementary: bool
        Write the complementary dataset to the output file.


    Returns
    -------
        dict: Merged dataset.
    """
    merged = {}
    merged_complementary = {} if write_complementary else None
    for dataset in datasets:
        for key, value in dataset.items():
            if key in merged:
                merged[key] += value[start:end:stride]
            else:
                merged[key] = value[start:end:stride]

            if write_complementary:
                end = end if end is not None else len(value)
                selected_indices = set(range(start, end, stride))
                total_indices = set(range(len(value)))
                complementary_indices = total_indices - selected_indices
                complementary_data = [value[i] for i in sorted(complementary_indices)]

                if key in merged_complementary:
                    merged_complementary[key] += complementary_data
                else:
                    merged_complementary[key] = complementary_data

    return merged, merged_complementary


def save_dataset(data: Dict[Any, Any], output_path: str) -> None:
    """
    Save a dataset to a pickle file.

    Parameters
    ----------
    data: dict
        Dataset to save.
    output_path: str
        Path to the output pickle file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Merge datasets from different sources into a single dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--start",
        type=int,
        default=0,
        help="Index of the first sample to include from each dataset.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Index of the last sample to include from each dataset.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for selecting samples from each dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="merged_ref_data.pkl",
        help="Path to the output dataset.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )

    parser.add_argument(
        "--write-complementary",
        action="store_true",
        help="Write the complementary dataset to the output file.",
    )

    args = parser.parse_args()

    # Decorative header
    header = r"""
╔════════════════════════════════════════════════════════════╗
║                       Merging Datasets                     ║
╚════════════════════════════════════════════════════════════╝
    """
    print(header)

    # Command-line arguments section
    print(" Command-line Arguments")
    print(" " + "─" * 58)
    for arg, value in vars(args).items():
        if arg == "files":
            print(f" {arg:<20}")
            for i, file in enumerate(value):
                print(f"   {i + 1:>3}. {file}")
        else:
            print(f" {arg:<20} {str(value):<45}")
    print(" " + "─" * 58)

    # Load the datasets
    try:
        datasets = [load_dataset(file) for file in args.files]
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return

    # Debugging details
    if args.debug:
        print("\n Debug: Dataset Details")
        print(" " + "═" * 58)
        for i, dataset in enumerate(datasets):
            print(f" Dataset {i + 1:>2} ({args.files[i]}):")
            for key, value in dataset.items():
                print(f" {key:<20} {len(value):>8}")
            print(" " + "═" * 58)

    # Merge the datasets
    merged_dataset, complementary_dataset = merge_datasets(
        datasets, args.start, args.end, args.stride, args.write_complementary
    )

    # Save the merged dataset
    try:
        save_dataset(merged_dataset, args.output)
        print(f"\n ✅ Merged dataset saved to '{args.output}'")

        print("\n Merged Dataset Keys and Lengths")
        print(" " + "─" * 58)
        for key, value in merged_dataset.items():
            print(f" {key:<20} {len(value):>8}")
        print(" " + "─" * 58)

        if args.write_complementary:
            complementary_output = args.output.replace(".pkl", "_complementary.pkl")
            save_dataset(complementary_dataset, complementary_output)
            print(f"\n ✅ Complementary dataset saved to '{complementary_output}'")

            print("\n Complementary Dataset Keys and Lengths")
            print(" " + "─" * 58)
            for key, value in complementary_dataset.items():
                print(f" {key:<20} {len(value):>8}")
            print(" " + "─" * 58)
    except Exception as e:
        print(f" ❌ Error saving merged dataset: {e}")

    footer = r"""
╔════════════════════════════════════════════════════════════╗
║                Datasets Successfully Merged!               ║
╚════════════════════════════════════════════════════════════╝
    """
    print(footer)


if __name__ == "__main__":
    main()
