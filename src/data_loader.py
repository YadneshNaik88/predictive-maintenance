"""
data_loader.py

Data loading utilities for the NASA C-MAPSS Turbofan Engine Degradation Dataset.
Phase 1: Loads raw C-MAPSS subset files into structured Pandas DataFrames with named columns.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Column definitions for C-MAPSS FD001 - FD004 datasets
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
SENSOR_COLUMNS = [f"sensor_measurement_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def get_default_data_dir() -> Path:
    """Resolve the default raw data directory relative to the repository root."""
    # Try finding data/raw relative to current file
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "data" / "raw"
    if candidate.exists():
        return candidate
    
    # Fallback to current working directory
    cwd_candidate = Path.cwd() / "data" / "raw"
    if cwd_candidate.exists():
        return cwd_candidate
    
    # Relative fallback for notebooks folder execution
    parent_candidate = Path.cwd().parent / "data" / "raw"
    if parent_candidate.exists():
        return parent_candidate

    return candidate


def load_cmapss(
    subset: str = "FD001",
    data_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load NASA C-MAPSS dataset files for a given subset (default: FD001).

    Parameters
    ----------
    subset : str, default="FD001"
        C-MAPSS dataset subset identifier (e.g., 'FD001', 'FD002', 'FD003', 'FD004').
    data_dir : str or Path, optional
        Path to directory containing raw .txt files. If None, auto-resolved.

    Returns
    -------
    train_df : pd.DataFrame
        Training set with 26 named columns: unit_number, time_in_cycles,
        3 operational settings, and 21 sensor measurements.
    test_df : pd.DataFrame
        Test set with the same 26 named columns.
    rul_df : pd.DataFrame
        True Remaining Useful Life (RUL) values for the test set engines.
    """
    raw_path = Path(data_dir) if data_dir is not None else get_default_data_dir()

    train_file = raw_path / f"train_{subset}.txt"
    test_file = raw_path / f"test_{subset}.txt"
    rul_file = raw_path / f"RUL_{subset}.txt"

    for file_path in [train_file, test_file, rul_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required C-MAPSS data file not found at: {file_path.resolve()}"
            )

    # C-MAPSS files are whitespace-delimited without headers
    train_df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        dtype={"unit_number": int, "time_in_cycles": int},
    )

    test_df = pd.read_csv(
        test_file,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        dtype={"unit_number": int, "time_in_cycles": int},
    )

    rul_df = pd.read_csv(
        rul_file,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        dtype={"RUL": int},
    )
    # Add 1-based unit_number to match test set unit IDs
    rul_df["unit_number"] = range(1, len(rul_df) + 1)
    rul_df = rul_df[["unit_number", "RUL"]]

    return train_df, test_df, rul_df


if __name__ == "__main__":
    print("Testing load_cmapss()...")
    train, test, rul = load_cmapss(subset="FD001")
    print(f"Train shape: {train.shape} (Units: {train['unit_number'].nunique()})")
    print(f"Test shape:  {test.shape} (Units: {test['unit_number'].nunique()})")
    print(f"RUL shape:   {rul.shape} (Units: {len(rul)})")
    print("\nFirst 3 rows of train_df:")
    print(train.head(3))
    print("\nFirst 3 rows of rul_df:")
    print(rul.head(3))
    print("\nData loading successful!")
