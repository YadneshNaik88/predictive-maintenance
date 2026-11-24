"""
uncertainty.py

Uncertainty Quantification and Prediction Intervals for RUL Regression.
Phase 4 & 4b:
- Heteroscedastic Binned-Residual Prediction Intervals (Default, empirically grounded)
- Monte Carlo Dropout (MC-Dropout) Epistemic Uncertainty Estimation (Baseline comparison)
- Uncertainty calibration persistence and unified agent inference interface.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.model import LSTMRegressor
from src.train import DEFAULT_TARGET_SCALE, unscale_target


# Robust resolution of project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "best_lstm_model.pt"
DEFAULT_MC_CALIB_PATH = PROJECT_ROOT / "outputs" / "uncertainty_calibration.json"
DEFAULT_RESIDUAL_CALIB_PATH = PROJECT_ROOT / "outputs" / "residual_interval_calibration.json"

DEFAULT_BIN_EDGES = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0]


def predict_point(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    target_scale: float = DEFAULT_TARGET_SCALE,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Perform standard deterministic point prediction in evaluation mode (dropout disabled).

    Parameters
    ----------
    model : nn.Module
        Trained LSTM model.
    X : np.ndarray or torch.Tensor
        Input sequences of shape (N, 30, 14) or (30, 14).
    target_scale : float, default=125.0
        Multiplier to unscale normalized model output [0, 1] to cycle units.
    device : torch.device, optional
        Hardware device.

    Returns
    -------
    y_pred : np.ndarray
        Predicted RUL point estimates in cycles, shape (N,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()

    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32)
    elif isinstance(X, torch.Tensor):
        if X.dim() == 2:
            X_tensor = X.unsqueeze(0).float()
        else:
            X_tensor = X.float()
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(X)}")

    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        preds_norm = model(X_tensor).cpu().numpy()
        if preds_norm.ndim == 0:
            preds_norm = np.array([preds_norm.item()])
        y_pred = unscale_target(preds_norm, target_scale=target_scale)

    return np.asarray(y_pred, dtype=np.float32)


# =============================================================================
# Binned Residual Interval Method (Phase 4b Default)
# =============================================================================

def fit_binned_residual_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_edges: Optional[List[float]] = None,
    alpha: float = 0.10,
    min_samples: int = 100,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Fit empirical residual prediction intervals binned by PREDICTED RUL using validation data.

    Computes signed residuals r = y_true - y_pred on validation sequences,
    bins by y_pred (which is known at inference time), and calculates the
    (alpha/2) and (1 - alpha/2) percentiles per bin (e.g. 5th and 95th for 90% interval).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth validation RUL values in cycles (e.g. capped at 125).
    y_pred : np.ndarray
        Validation point predictions in cycles.
    bin_edges : list of float, optional
        Edges defining prediction bins. Defaults to [0.0, 25.0, 50.0, 75.0, 100.0, 125.0].
    alpha : float, default=0.10
        Significance level (alpha=0.10 gives nominal 90% coverage).
    min_samples : int, default=100
        Minimum validation sample count required per bin.
    save_path : str or Path, optional
        Target file path to save calibration JSON. Defaults to outputs/residual_interval_calibration.json.

    Returns
    -------
    calib_data : dict
        Dictionary containing bin edges, quantiles, sample counts, and validation metadata.
    """
    if bin_edges is None:
        bin_edges = list(DEFAULT_BIN_EDGES)

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    residuals = y_true - y_pred

    q_low_p = (alpha / 2.0) * 100.0
    q_high_p = (1.0 - alpha / 2.0) * 100.0

    # Digitize predictions based on inner edges
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    bins_list = []
    num_bins = len(bin_edges) - 1

    for b in range(num_bins):
        low_e = float(bin_edges[b])
        high_e = float(bin_edges[b + 1])
        mask = (bin_indices == b)
        count = int(np.sum(mask))

        if count < min_samples:
            print(f"[Warning] Bin {b} [{low_e}, {high_e}) has {count} samples (< {min_samples}).")

        res_b = residuals[mask] if count > 0 else residuals
        q05 = float(np.percentile(res_b, q_low_p)) if count > 0 else 0.0
        q95 = float(np.percentile(res_b, q_high_p)) if count > 0 else 0.0

        bins_list.append({
            "bin_idx": b,
            "edge_lower": low_e,
            "edge_upper": high_e,
            "count": count,
            "mean_residual": float(np.mean(res_b)) if count > 0 else 0.0,
            "std_residual": float(np.std(res_b)) if count > 0 else 0.0,
            "q05": q05,
            "q95": q95,
            "raw_width": float(q95 - q05),
        })

    # Overall validation coverage verification
    low_val = np.zeros_like(y_pred)
    up_val = np.zeros_like(y_pred)
    for i, p in enumerate(y_pred):
        b = bin_indices[i]
        low_val[i] = max(0.0, min(125.0, p + bins_list[b]["q05"]))
        up_val[i] = min(125.0, max(0.0, p + bins_list[b]["q95"]))

    val_coverage = float(np.mean((y_true >= low_val) & (y_true <= up_val)))

    calib_data = {
        "method": "binned_residual",
        "target_nominal_level": float(1.0 - alpha),
        "bin_edges": [float(e) for e in bin_edges],
        "num_val_samples": len(y_true),
        "validation_empirical_coverage": round(val_coverage, 4),
        "bins": bins_list,
    }

    if save_path is None:
        save_path = DEFAULT_RESIDUAL_CALIB_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=4)

    return calib_data


def binned_residual_interval(
    y_pred: Union[np.ndarray, float],
    calib_data: Optional[Dict[str, Any]] = None,
    calib_path: Optional[Union[str, Path]] = None,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Construct heteroscedastic prediction intervals for predicted RUL using binned residual quantiles.

    Calculates:
        lower = clip(y_pred + q05_bin, 0.0, 125.0)
        upper = clip(y_pred + q95_bin, 0.0, 125.0)

    Parameters
    ----------
    y_pred : np.ndarray or float
        Predicted RUL point estimates in cycles.
    calib_data : dict, optional
        Pre-loaded calibration dictionary.
    calib_path : str or Path, optional
        Path to residual_interval_calibration.json.

    Returns
    -------
    lower : np.ndarray or float
        Lower bound of the prediction interval, clipped in [0.0, 125.0].
    upper : np.ndarray or float
        Upper bound of the prediction interval, clipped in [0.0, 125.0].
    """
    if calib_data is None:
        if calib_path is None:
            calib_path = DEFAULT_RESIDUAL_CALIB_PATH
        calib_path = Path(calib_path)
        if not calib_path.exists() and (PROJECT_ROOT / calib_path).exists():
            calib_path = PROJECT_ROOT / calib_path

        if not calib_path.exists():
            raise FileNotFoundError(f"Residual calibration file not found at: {calib_path.resolve()}")

        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

    bin_edges = calib_data.get("bin_edges", DEFAULT_BIN_EDGES)
    bins_info = calib_data["bins"]

    is_scalar = isinstance(y_pred, (int, float)) or (isinstance(y_pred, np.ndarray) and y_pred.ndim == 0)
    preds_arr = np.atleast_1d(np.asarray(y_pred, dtype=np.float32))

    # Inner edges for digitize
    inner_edges = bin_edges[1:-1]
    bin_indices = np.digitize(preds_arr, inner_edges)
    bin_indices = np.clip(bin_indices, 0, len(bins_info) - 1)

    lowers = np.zeros_like(preds_arr)
    uppers = np.zeros_like(preds_arr)

    for i, p in enumerate(preds_arr):
        b = bin_indices[i]
        q05 = bins_info[b]["q05"]
        q95 = bins_info[b]["q95"]
        lowers[i] = max(0.0, min(125.0, float(p + q05)))
        uppers[i] = min(125.0, max(0.0, float(p + q95)))

    if is_scalar:
        return float(lowers[0]), float(uppers[0])

    return lowers, uppers


# =============================================================================
# MC-Dropout Epistemic Uncertainty (Phase 4 Baseline)
# =============================================================================

def mc_dropout_predict(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    n_samples: int = 100,
    seed: int = 42,
    target_scale: float = DEFAULT_TARGET_SCALE,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo Dropout stochastic forward passes to estimate predictive uncertainty.

    Keeps the LSTM backbone in evaluation mode while activating ONLY the post-LSTM
    dropout layer via `model.enable_mc_dropout()`. Reconstructs predictions in
    unscaled true RUL cycles.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch LSTM model with `enable_mc_dropout()` and `disable_mc_dropout()`.
    X : np.ndarray or torch.Tensor
        Input feature sequences of shape (N, seq_len, num_features) or (seq_len, num_features).
    n_samples : int, default=100
        Number of stochastic MC-Dropout forward passes.
    seed : int, default=42
        Random seed for deterministic, reproducible sampling.
    target_scale : float, default=125.0
        Multiplier to unscale normalized model output [0, 1] into original cycle units.
    device : torch.device, optional
        Hardware device for computation. Defaults to CUDA if available, else CPU.

    Returns
    -------
    mean_rul : np.ndarray
        Mean predicted RUL in cycles across all MC samples, shape (N,).
    std_rul : np.ndarray
        Standard deviation (uncertainty) of predicted RUL in cycles, shape (N,).
    raw_samples : np.ndarray
        All individual stochastic MC predictions in cycles, shape (n_samples, N).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()

    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32)
    elif isinstance(X, torch.Tensor):
        if X.dim() == 2:
            X_tensor = X.unsqueeze(0).float()
        else:
            X_tensor = X.float()
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(X)}")

    X_tensor = X_tensor.to(device)
    batch_size = X_tensor.shape[0]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    raw_samples_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            preds_norm = model(X_tensor).cpu().numpy()
            if preds_norm.ndim == 0:
                preds_norm = np.array([preds_norm.item()])
            preds_cycles = unscale_target(preds_norm, target_scale=target_scale)
            raw_samples_list.append(preds_cycles)

    if hasattr(model, "disable_mc_dropout"):
        model.disable_mc_dropout()
    model.eval()

    raw_samples = np.array(raw_samples_list, dtype=np.float32)
    mean_rul = np.mean(raw_samples, axis=0)
    std_rul = np.std(raw_samples, axis=0, ddof=1) if n_samples > 1 else np.zeros(batch_size, dtype=np.float32)

    return mean_rul, std_rul, raw_samples


def interval_from_samples(
    mean: Union[np.ndarray, float],
    std: Union[np.ndarray, float],
    level: float = 0.90,
    k: float = 1.0,
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Construct symmetric calibrated prediction intervals from MC-Dropout mean and std.

    Calculates:
        lower = max(0.0, mean - z * k * std)
        upper = min(125.0, mean + z * k * std)

    Parameters
    ----------
    mean : np.ndarray or float
        Predicted mean RUL in cycles.
    std : np.ndarray or float
        Predicted standard deviation in cycles.
    level : float, default=0.90
        Nominal confidence level in (0, 1).
    k : float, default=1.0
        Calibration scaling factor.

    Returns
    -------
    lower : np.ndarray or float
        Lower bound of prediction interval (>= 0 cycles).
    upper : np.ndarray or float
        Upper bound of prediction interval (<= 125 cycles).
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"Nominal confidence level must be in (0, 1), got {level}")

    z = float(stats.norm.ppf((1.0 + level) / 2.0))
    margin = z * k * std

    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        lower = max(0.0, float(mean - margin))
        upper = float(mean + margin)
    else:
        mean_arr = np.asarray(mean, dtype=np.float32)
        margin_arr = np.asarray(margin, dtype=np.float32)
        lower = np.maximum(0.0, mean_arr - margin_arr)
        upper = mean_arr + margin_arr

    return lower, upper


def fit_calibration_k(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    level: float = 0.90,
) -> float:
    """Fit a single scalar multiplier k using validation data for MC-Dropout."""
    y_true = np.asarray(y_true, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    z = float(stats.norm.ppf((1.0 + level) / 2.0))
    safe_std = np.maximum(std, 1e-6)
    required_k = np.abs(y_true - mean) / (z * safe_std)
    k_fitted = float(np.quantile(required_k, level))
    return max(1.0, k_fitted)


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical coverage fraction of ground truth targets within [lower, upper]."""
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    levels: Tuple[float, ...] = (0.50, 0.80, 0.90),
    k: float = 1.0,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute comprehensive MC-Dropout uncertainty metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    abs_error = np.abs(mean - y_true)

    spearman_corr, spearman_pvalue = stats.spearmanr(std, abs_error)

    metrics: Dict[str, Union[float, Dict[str, float]]] = {
        "spearman_correlation": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "spearman_pvalue": float(spearman_pvalue) if not np.isnan(spearman_pvalue) else 1.0,
        "mean_std": float(np.mean(std)),
        "std_std": float(np.std(std)),
        "coverage": {},
        "mean_width": {},
    }

    for lvl in levels:
        lower, upper = interval_from_samples(mean, std, level=lvl, k=k)
        cov = compute_coverage(y_true, lower, upper)
        width = float(np.mean(upper - lower))
        metrics["coverage"][f"{int(lvl*100)}%"] = cov
        metrics["mean_width"][f"{int(lvl*100)}%"] = width

    return metrics


def load_uncertainty_model_and_calib(
    checkpoint_path: Optional[Union[str, Path]] = None,
    calib_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[LSTMRegressor, float, float]:
    """Load saved PyTorch LSTM model checkpoint and calibrated uncertainty scaling factor."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() and (PROJECT_ROOT / checkpoint_path).exists():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path.resolve()}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})

    input_size = cfg.get("input_size", 14)
    hidden_size = cfg.get("hidden_size", 64)
    num_layers = cfg.get("num_layers", 2)
    dropout_rate = cfg.get("dropout_rate", 0.2)
    target_scale = cfg.get("target_scale", DEFAULT_TARGET_SCALE)

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if calib_path is None:
        calib_path = DEFAULT_MC_CALIB_PATH
    calib_path = Path(calib_path)
    if not calib_path.exists() and (PROJECT_ROOT / calib_path).exists():
        calib_path = PROJECT_ROOT / calib_path

    k_factor = 1.0
    if calib_path.exists():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
                k_factor = float(calib_data.get("k_factor", 1.0))
        except Exception:
            k_factor = 1.0

    return model, k_factor, target_scale


# =============================================================================
# Unified Agent Inference API (Default: Binned Residual)
# =============================================================================

def predict_rul_with_interval(
    sequence_or_batch: Union[np.ndarray, torch.Tensor],
    model: Optional[LSTMRegressor] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    calib_path: Optional[Union[str, Path]] = None,
    level: float = 0.90,
    n_samples: int = 100,
    k: Optional[float] = None,
    method: str = "binned_residual",
    device: Optional[torch.device] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Unified entry point for RUL prediction with uncertainty intervals.

    Designed for seamless integration with downstream agents (e.g. Prognostic Agent).
    By default, employs the heteroscedastic **Binned-Residual Interval** method (Phase 4b),
    which provides sharp, degradation-dependent intervals. Alternatively supports
    Monte Carlo Dropout (`method="mc_dropout"`).

    Parameters
    ----------
    sequence_or_batch : np.ndarray or torch.Tensor
        Input sensor telemetry sequence(s). Supports:
        - Single engine window: shape (30, 14)
        - Single engine batch:  shape (1, 30, 14)
        - Multi-engine batch:   shape (N, 30, 14)
    model : LSTMRegressor, optional
        Pre-loaded model instance. If None, loaded automatically from `checkpoint_path`.
    checkpoint_path : str or Path, optional
        Path to best_lstm_model.pt. Defaults to outputs/best_lstm_model.pt.
    calib_path : str or Path, optional
        Path to calibration JSON. Defaults to residual_interval_calibration.json (for binned_residual)
        or uncertainty_calibration.json (for mc_dropout).
    level : float, default=0.90
        Nominal prediction interval confidence level (0.90 for 90% confidence).
    n_samples : int, default=100
        Number of Monte Carlo forward passes (used only if method='mc_dropout').
    k : float, optional
        Manual calibration factor override (used only if method='mc_dropout').
    method : {'binned_residual', 'mc_dropout'}, default='binned_residual'
        Uncertainty quantification method.
    device : torch.device, optional
        Hardware device for inference.

    Returns
    -------
    mean : float or np.ndarray
        Predicted RUL point estimate in cycles. Returns float for single-sequence input,
        or np.ndarray of shape (N,) for batch input.
    lower : float or np.ndarray
        Lower bound of prediction interval in cycles (clamped >= 0).
    upper : float or np.ndarray
        Upper bound of prediction interval in cycles (clamped <= 125 for binned_residual).

    Example
    -------
    >>> from src.uncertainty import predict_rul_with_interval
    >>> import numpy as np
    >>> window = np.random.randn(30, 14).astype(np.float32)
    >>> mean_rul, lower_rul, upper_rul = predict_rul_with_interval(window, method="binned_residual")
    >>> print(f"Predicted RUL: {mean_rul:.1f} cycles [90% CI: {lower_rul:.1f} - {upper_rul:.1f}]")
    """
    if method not in ("binned_residual", "mc_dropout"):
        raise ValueError(f"Unknown uncertainty method: '{method}'. Expected 'binned_residual' or 'mc_dropout'.")

    target_scale = DEFAULT_TARGET_SCALE

    if model is None:
        model, loaded_k, target_scale = load_uncertainty_model_and_calib(
            checkpoint_path=checkpoint_path,
            calib_path=calib_path if method == "mc_dropout" else None,
            device=device,
        )
        if k is None:
            k = loaded_k

    # Ensure 3D array
    is_single_sequence = False
    if isinstance(sequence_or_batch, np.ndarray):
        if sequence_or_batch.ndim == 2:
            is_single_sequence = True
            X_in = np.expand_dims(sequence_or_batch, axis=0)
        elif sequence_or_batch.ndim == 3:
            is_single_sequence = (sequence_or_batch.shape[0] == 1)
            X_in = sequence_or_batch
        else:
            raise ValueError(f"Expected 2D or 3D numpy array, got shape {sequence_or_batch.shape}")
    elif isinstance(sequence_or_batch, torch.Tensor):
        if sequence_or_batch.dim() == 2:
            is_single_sequence = True
            X_in = sequence_or_batch.unsqueeze(0).cpu().numpy()
        elif sequence_or_batch.dim() == 3:
            is_single_sequence = (sequence_or_batch.shape[0] == 1)
            X_in = sequence_or_batch.cpu().numpy()
        else:
            raise ValueError(f"Expected 2D or 3D torch.Tensor, got shape {sequence_or_batch.shape}")
    else:
        raise TypeError(f"Unsupported input type: {type(sequence_or_batch)}")

    if method == "binned_residual":
        # 1. Deterministic point prediction in eval mode
        y_pred = predict_point(model, X_in, target_scale=target_scale, device=device)
        # 2. Binned residual interval bounds
        res_calib_path = calib_path if calib_path is not None else DEFAULT_RESIDUAL_CALIB_PATH
        lower_arr, upper_arr = binned_residual_interval(y_pred, calib_path=res_calib_path)
        mean_arr = y_pred
    else:
        # MC-Dropout stochastic sampling
        mean_arr, std_arr, _ = mc_dropout_predict(
            model=model,
            X=X_in,
            n_samples=n_samples,
            target_scale=target_scale,
            device=device,
        )
        lower_arr, upper_arr = interval_from_samples(
            mean=mean_arr,
            std=std_arr,
            level=level,
            k=k if k is not None else 1.0,
        )

    if is_single_sequence and len(mean_arr) == 1:
        return float(mean_arr[0]), float(lower_arr[0]), float(upper_arr[0])

    return mean_arr, lower_arr, upper_arr
