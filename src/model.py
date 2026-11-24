"""
model.py

PyTorch LSTM Neural Network for Remaining Useful Life (RUL) Regression.
Phase 3: Deep sequence modeling on NASA C-MAPSS multivariate sensor telemetry.
Designed with post-LSTM Dropout support for Monte Carlo Dropout (Phase 4).
"""

from typing import Optional
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    LSTM-based Neural Network for turbofan Remaining Useful Life (RUL) estimation.

    Architecture:
        Input Sequence (batch, seq_len, num_features)
             ↓
        Multi-layer Stacked LSTM
             ↓
        Final Timestep Hidden State (last cycle of temporal window)
             ↓
        Dropout Layer (can be forced active at inference for MC-Dropout)
             ↓
        Fully Connected Linear Layer (hidden_size -> 1)
             ↓
        Predicted RUL (scalar per sequence)

    Parameters
    ----------
    input_size : int, default=14
        Number of input sensor features per timestep.
    hidden_size : int, default=64
        Number of hidden units in each LSTM layer.
    num_layers : int, default=2
        Number of stacked recurrent LSTM layers.
    dropout_rate : float, default=0.2
        Dropout probability applied after the recurrent layer.
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Explicit Post-LSTM Dropout (Crucial for Phase 4 MC-Dropout at inference)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output linear regression head
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, sequence_length, num_features)

        Returns
        -------
        torch.Tensor of shape (batch_size,) containing predicted RUL.
        """
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Extract representation at the final timestep of the sequence
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_hidden)

        # Project to single RUL scalar
        out = self.fc(dropped)

        return out.squeeze(-1)

    def enable_mc_dropout(self) -> None:
        """
        Enable Monte Carlo Dropout during evaluation/inference mode.
        Keeps dropout active while retaining eval mode for normalization/other layers.
        """
        self.dropout.train()

    def disable_mc_dropout(self) -> None:
        """Disable MC Dropout and restore standard evaluation behavior."""
        self.dropout.eval()


if __name__ == "__main__":
    print("Testing LSTMRegressor...")
    model = LSTMRegressor(input_size=14, hidden_size=64, num_layers=2, dropout_rate=0.2)
    sample_input = torch.randn(16, 30, 14)
    sample_out = model(sample_input)
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {sample_out.shape}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test MC-Dropout toggle
    model.eval()
    print(f"Default eval mode dropout training: {model.dropout.training}")
    model.enable_mc_dropout()
    print(f"After enable_mc_dropout():         {model.dropout.training}")
    model.disable_mc_dropout()
    print(f"After disable_mc_dropout():        {model.dropout.training}")
    print("LSTMRegressor implementation verified successfully!")
