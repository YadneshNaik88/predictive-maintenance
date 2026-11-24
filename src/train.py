"""
train.py

Training and evaluation pipeline for the PyTorch LSTM RUL Regressor.
Includes target normalization helpers, validation MAE-based early stopping,
standard regression metrics (MAE, RMSE), and NASA's official asymmetric C-MAPSS scoring function.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DEFAULT_TARGET_SCALE: float = 125.0


def scale_target(
    y: Union[np.ndarray, torch.Tensor, float],
    target_scale: float = DEFAULT_TARGET_SCALE,
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Normalize Remaining Useful Life (RUL) target to [0, 1] range for gradient stability.

    Parameters
    ----------
    y : np.ndarray, torch.Tensor, or float
        RUL targets in cycle units (e.g. 0 to 125).
    target_scale : float, default=125.0
        Maximum RUL scale factor.

    Returns
    -------
    Normalized target values in [0, 1].
    """
    return y / target_scale


def unscale_target(
    y_scaled: Union[np.ndarray, torch.Tensor, float],
    target_scale: float = DEFAULT_TARGET_SCALE,
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Convert normalized model predictions [0, 1] back into original RUL cycles.

    Parameters
    ----------
    y_scaled : np.ndarray, torch.Tensor, or float
        Normalized predictions.
    target_scale : float, default=125.0
        Maximum RUL scale factor.

    Returns
    -------
    Predictions in true RUL cycle units.
    """
    return y_scaled * target_scale


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE) in cycle units."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) in cycle units."""
    return float(root_mean_squared_error(y_true, y_pred))


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute NASA Ames official asymmetric C-MAPSS scoring function.

    The score penalizes late predictions (overestimating RUL, risking catastrophic failure)
    more heavily than early predictions (underestimating RUL, causing premature maintenance).

    For error d_i = y_pred_i - y_true_i:
        s_i = exp(-d_i / 13) - 1   if d_i < 0  (early prediction)
        s_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction)
        Score = sum(s_i)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth Remaining Useful Life values in cycle units.
    y_pred : np.ndarray
        Predicted Remaining Useful Life values in cycle units.

    Returns
    -------
    float
        Cumulative asymmetric NASA penalty score.
    """
    d = np.array(y_pred, dtype=float) - np.array(y_true, dtype=float)
    # Clip large positive errors to prevent exponential overflow in extreme edge cases
    d_clipped = np.clip(d, -100.0, 50.0)
    score = np.where(
        d_clipped < 0,
        np.exp(-d_clipped / 13.0) - 1.0,
        np.exp(d_clipped / 10.0) - 1.0,
    )
    return float(np.sum(score))


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    target_scale: float = DEFAULT_TARGET_SCALE,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate trained model on a PyTorch DataLoader, returning unscaled predictions and metrics.

    Parameters
    ----------
    model : nn.Module
        PyTorch model predicting normalized RUL in [0, 1].
    data_loader : DataLoader
        DataLoader providing (batch_x, batch_y) or batch_x.
    target_scale : float, default=125.0
        Scale factor used to convert normalized predictions to true RUL cycles.
    device : torch.device, optional
        Computation device (CPU/CUDA).

    Returns
    -------
    y_pred : np.ndarray
        Model predictions in true RUL cycle units.
    y_true : np.ndarray
        Ground truth targets in true RUL cycle units.
    metrics : dict
        Dictionary of computed metrics in cycle units (MAE, RMSE, NASA_Score).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    preds_list: List[float] = []
    targets_list: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)
                pred_norm = model(batch_x)
                # Unscale prediction to RUL cycles
                pred_cycles = pred_norm * target_scale
                preds_list.extend(pred_cycles.cpu().numpy().tolist())

                # If batch_y was provided in normalized [0, 1] range, unscale it
                y_np = batch_y.cpu().numpy()
                if y_np.size > 0 and np.max(y_np) <= 1.5:
                    y_cycles = y_np * target_scale
                else:
                    y_cycles = y_np
                targets_list.extend(y_cycles.tolist())
            else:
                batch_x = batch.to(device)
                pred_norm = model(batch_x)
                pred_cycles = pred_norm * target_scale
                preds_list.extend(pred_cycles.cpu().numpy().tolist())

    y_pred = np.array(preds_list, dtype=np.float32)
    y_true = np.array(targets_list, dtype=np.float32) if targets_list else np.array([])

    metrics: Dict[str, float] = {}
    if len(y_true) > 0:
        metrics["MAE"] = compute_mae(y_true, y_pred)
        metrics["RMSE"] = compute_rmse(y_true, y_pred)
        metrics["NASA_Score"] = compute_nasa_score(y_true, y_pred)

    return y_pred, y_true, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 0.001,
    patience: int = 8,
    target_scale: float = DEFAULT_TARGET_SCALE,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, Union[List[float], int, float]]]:
    """
    Train LSTM model with target normalization, Adam optimizer, and Early Stopping on Validation MAE.

    Parameters
    ----------
    model : nn.Module
        PyTorch LSTM model instance.
    train_loader : DataLoader
        DataLoader providing normalized training sequences (batch_x, batch_y_norm).
    val_loader : DataLoader
        DataLoader providing normalized validation sequences (batch_x, batch_y_norm).
    epochs : int, default=60
        Maximum training epochs.
    lr : float, default=0.001
        Learning rate for Adam optimizer.
    patience : int, default=8
        Epochs with no Validation MAE improvement before early stopping.
    target_scale : float, default=125.0
        Scale factor used to convert normalized predictions to true RUL cycles.
    device : torch.device, optional
        Target hardware device.
    verbose : bool, default=False
        Print epoch progress if True.

    Returns
    -------
    model : nn.Module
        Model restored to best validation MAE epoch weights.
    history : dict
        Training history containing train_loss, val_loss, val_mae, best_epoch, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_maes: List[float] = []

    for epoch in range(1, epochs + 1):
        # 1. Training pass
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)

        # 2. Validation pass
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)

                running_val_loss += loss.item()
                val_batches += 1

                val_preds_all.extend((preds * target_scale).cpu().numpy().tolist())
                val_targets_all.extend((batch_y * target_scale).cpu().numpy().tolist())

        avg_val_loss = running_val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        val_mae = compute_mae(np.array(val_targets_all), np.array(val_preds_all))
        val_maes.append(val_mae)

        # 3. Check early stopping on VALIDATION MAE only (Brief requirement)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = avg_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (Best epoch: {best_epoch} with Val MAE: {best_val_mae:.4f})")
                break

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.2f} cycles")

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_mae": val_maes,
        "best_epoch": best_epoch,
        "epochs_trained": len(train_losses),
        "best_val_mae": best_val_mae,
        "best_val_loss": best_val_loss,
    }

    return model, history
