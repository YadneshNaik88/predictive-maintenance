"""
preprocessing.py

Phase 2 Preprocessing Pipeline for NASA C-MAPSS Turbofan Degradation (FD001).
Implements feature screening (dropping uninformative columns), piecewise-linear
RUL calculation and capping, zero-leakage MinMax feature scaling, and rolling-window
temporal sequence generation for train and test sets.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 10 Uninformative columns confirmed from Phase 1 EDA (zero/near-zero variance)
UNINFORMATIVE_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_measurement_1",
    "sensor_measurement_5",
    "sensor_measurement_6",
    "sensor_measurement_10",
    "sensor_measurement_16",
    "sensor_measurement_18",
    "sensor_measurement_19",
]

# 14 Retained informative sensor columns
RETAINED_SENSOR_COLUMNS = [
    f"sensor_measurement_{i}"
    for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]

# Default scaler save location
DEFAULT_SCALER_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "scaler.pkl"
)


def drop_uninformative_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop zero-variance and uninformative feature columns from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (train or test).
    columns_to_drop : list of str, optional
        List of column names to drop. Defaults to the 10 uninformative columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed.
    """
    if columns_to_drop is None:
        columns_to_drop = UNINFORMATIVE_COLUMNS

    cols_present = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=cols_present).copy()


def compute_rul(
    df: pd.DataFrame,
    max_rul: int = 125,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Compute ground-truth Remaining Useful Life (RUL) and piecewise-linear capped RUL.

    For run-to-failure training data, true RUL is computed per row as:
        RUL = max_cycles_for_that_engine - current_cycle
    Piecewise-linear capping sets:
        RUL_capped = min(RUL, max_rul)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'unit_number' and 'time_in_cycles'.
    max_rul : int, default=125
        Threshold above which RUL is capped to represent the healthy operating state.
    is_train : bool, default=True
        Flag indicating if the dataset is training data (run-to-failure).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'RUL' and 'RUL_capped' columns (if is_train=True).
    """
    df_out = df.copy()
    if not is_train:
        return df_out

    # Compute maximum cycle reached by each engine unit
    max_cycles = (
        df_out.groupby("unit_number")["time_in_cycles"]
        .max()
        .reset_index(name="max_cycles")
    )
    df_out = df_out.merge(max_cycles, on="unit_number", how="left")

    # True linear RUL
    df_out["RUL"] = df_out["max_cycles"] - df_out["time_in_cycles"]

    # Piecewise-linear capped RUL
    df_out["RUL_capped"] = np.minimum(df_out["RUL"], max_rul)

    # Drop temporary max_cycles column
    df_out = df_out.drop(columns=["max_cycles"])

    return df_out


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    scaler_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fit a MinMaxScaler [0, 1] on training sensor features and transform both train and test.

    Strictly avoids data leakage by fitting solely on training features. Saves the fitted
    scaler object to disk for deployment and model inference.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame.
    test_df : pd.DataFrame
        Test DataFrame.
    feature_cols : list of str, optional
        Features to scale. Defaults to RETAINED_SENSOR_COLUMNS.
    scaler_path : str or Path, optional
        Filepath to persist the fitted scaler. Defaults to outputs/scaler.pkl.

    Returns
    -------
    scaled_train_df : pd.DataFrame
        Training DataFrame with normalized feature columns.
    scaled_test_df : pd.DataFrame
        Test DataFrame with normalized feature columns.
    scaler : MinMaxScaler
        Fitted Scikit-Learn MinMaxScaler object.
    """
    if feature_cols is None:
        feature_cols = [col for col in RETAINED_SENSOR_COLUMNS if col in train_df.columns]

    scaled_train_df = train_df.copy()
    scaled_test_df = test_df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    scaled_test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Save fitted scaler
    target_path = Path(scaler_path) if scaler_path is not None else DEFAULT_SCALER_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, target_path)

    return scaled_train_df, scaled_test_df, scaler


def generate_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "RUL_capped",
    return_unit_ids: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate rolling temporal window sequences (X, y) for training sequence models.

    For each engine, slides a window of `sequence_length` cycles.
    Engines with fewer total cycles than `sequence_length` are left-padded
    with the first observed cycle repeated.

    Parameters
    ----------
    df : pd.DataFrame
        Input training DataFrame with unit_number, time_in_cycles, features, and label_col.
    sequence_length : int, default=30
        Length of the temporal window.
    feature_cols : list of str, optional
        Feature column names. Defaults to RETAINED_SENSOR_COLUMNS.
    label_col : str, default="RUL_capped"
        Target label column name.
    return_unit_ids : bool, default=False
        If True, also returns an array of unit_number IDs for each generated sequence.

    Returns
    -------
    X : np.ndarray
        Feature sequences of shape [num_sequences, sequence_length, num_features].
    y : np.ndarray
        Target RUL values at the last cycle of each sequence, shape [num_sequences].
    unit_ids : np.ndarray (optional)
        Engine unit_number ID corresponding to each sequence.
    """
    if feature_cols is None:
        feature_cols = [col for col in RETAINED_SENSOR_COLUMNS if col in df.columns]

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    unit_id_list: List[int] = []

    for unit_id, group in df.groupby("unit_number"):
        group_features = group[feature_cols].values
        group_labels = group[label_col].values
        num_cycles = len(group)

        if num_cycles < sequence_length:
            # Left-pad by repeating the first row
            pad_len = sequence_length - num_cycles
            pad_matrix = np.repeat(group_features[0:1, :], pad_len, axis=0)
            X_seq = np.vstack([pad_matrix, group_features])
            X_list.append(X_seq)
            y_list.append(group_labels[-1])
            unit_id_list.append(int(unit_id))
        else:
            # Sliding window of sequence_length
            for end_idx in range(sequence_length, num_cycles + 1):
                start_idx = end_idx - sequence_length
                X_list.append(group_features[start_idx:end_idx, :])
                y_list.append(group_labels[end_idx - 1])
                unit_id_list.append(int(unit_id))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if return_unit_ids:
        unit_ids = np.array(unit_id_list, dtype=int)
        return X, y, unit_ids

    return X, y


def get_engine_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    sequence_length: int = 30,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "RUL_capped",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset at the ENGINE LEVEL to avoid sliding-window temporal leakage.

    Randomly partitions unique unit_numbers into train and validation sets,
    then generates sequences independently for each partition.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame with unit_number, time_in_cycles, features, and label_col.
    val_ratio : float, default=0.2
        Fraction of engine units held out for validation (e.g., 0.2 = 20 engines).
    sequence_length : int, default=30
        Length of the sliding window.
    feature_cols : list of str, optional
        Sensor feature names.
    label_col : str, default="RUL_capped"
        Target column.
    seed : int, default=42
        Random seed for reproducible engine splitting.

    Returns
    -------
    X_train : np.ndarray
        Training feature sequences [num_train_seqs, sequence_length, num_features].
    y_train : np.ndarray
        Training target RUL values [num_train_seqs].
    X_val : np.ndarray
        Validation feature sequences [num_val_seqs, sequence_length, num_features].
    y_val : np.ndarray
        Validation target RUL values [num_val_seqs].
    train_units : np.ndarray
        Array of engine IDs assigned to training.
    val_units : np.ndarray
        Array of engine IDs assigned to validation.
    """
    unique_units = np.sort(df["unit_number"].unique())
    num_val_engines = int(round(len(unique_units) * val_ratio))
    
    rng = np.random.default_rng(seed)
    shuffled_units = rng.permutation(unique_units)
    
    val_units = np.sort(shuffled_units[:num_val_engines])
    train_units = np.sort(shuffled_units[num_val_engines:])

    train_df_split = df[df["unit_number"].isin(train_units)]
    val_df_split = df[df["unit_number"].isin(val_units)]

    X_train, y_train = generate_sequences(
        train_df_split,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
        label_col=label_col,
    )
    X_val, y_val = generate_sequences(
        val_df_split,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
        label_col=label_col,
    )

    return X_train, y_train, X_val, y_val, train_units, val_units


def get_test_sequences(
    test_df: pd.DataFrame,
    sequence_length: int = 30,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the final evaluation sequence for each test engine.

    Generates exactly ONE sequence per test engine: the last `sequence_length`
    cycles of that engine (left-padded if cycles < sequence_length).

    Parameters
    ----------
    test_df : pd.DataFrame
        Test DataFrame containing unit_number and feature columns.
    sequence_length : int, default=30
        Length of the temporal window.
    feature_cols : list of str, optional
        Feature column names. Defaults to RETAINED_SENSOR_COLUMNS.

    Returns
    -------
    X_test : np.ndarray
        Test feature sequences of shape [num_test_engines, sequence_length, num_features].
    unit_order : np.ndarray
        Array of unit_number IDs matching the order of X_test sequences.
    """
    if feature_cols is None:
        feature_cols = [col for col in RETAINED_SENSOR_COLUMNS if col in test_df.columns]

    X_test_list: List[np.ndarray] = []
    unit_order_list: List[int] = []

    for unit_id, group in test_df.groupby("unit_number"):
        group_features = group[feature_cols].values
        num_cycles = len(group)
        unit_order_list.append(int(unit_id))

        if num_cycles < sequence_length:
            # Left-pad with the first row
            pad_len = sequence_length - num_cycles
            pad_matrix = np.repeat(group_features[0:1, :], pad_len, axis=0)
            X_seq = np.vstack([pad_matrix, group_features])
            X_test_list.append(X_seq)
        else:
            # Take the final sequence_length cycles
            X_test_list.append(group_features[-sequence_length:, :])

    X_test = np.array(X_test_list, dtype=np.float32)
    unit_order = np.array(unit_order_list, dtype=int)

    return X_test, unit_order


def count_padded_engines(
    df: pd.DataFrame,
    sequence_length: int = 30,
) -> int:
    """Helper to count how many engines have fewer total cycles than sequence_length."""
    lengths = df.groupby("unit_number")["time_in_cycles"].max()
    return int((lengths < sequence_length).sum())
