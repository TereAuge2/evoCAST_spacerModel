"""PyTorch module definitions for MLP and 1D CNN regressors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def assert_flat_dna_onehot_dim(n_features: int) -> None:
    """CNN path requires flat length L×4 (one position × four nucleotide channels)."""
    d = int(n_features)
    if d <= 0 or d % 4 != 0:
        raise ValueError(
            "CNN expects flat one-hot features with dimension divisible by 4 "
            f"(sequence length × 4 nucleotides ACGT); got n_features={n_features}."
        )


def to_channel_first(X: np.ndarray) -> np.ndarray:
    """
    Reshape flat one-hot (n, 4*L) to Conv1d layout (n, 4, L): four channels = A,C,G,T.

    There is no 1×d fallback: use MLP or a different embedding if the input is not DNA one-hot.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    assert_flat_dna_onehot_dim(d)
    L = d // 4
    return X.reshape(n, L, 4).transpose(0, 2, 1)


class MLPRegressorNet(nn.Module):
    """
    Fully connected regressor on a single vector per example (after scaling).

    Architecture: for hidden sizes ``(h1, h2, …, hk)``,
    ``Linear(in_dim→h1) → ReLU → Dropout → … → Linear(hk→1)``.
    The last layer outputs one scalar; training uses MSE vs the target (scaled in the pipeline).
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNN1DRegressorNet(nn.Module):
    """
    1D CNN on DNA one-hot with **4 input channels** (A, C, G, T) and length L (positions).

    **Forward path (tensor shape ``(batch, 4, L)``)**:
    - ``Conv1d(4 → C1, kernel k1, padding k1//2) → ReLU``
    - ``Conv1d(C1 → C2, kernel k2, padding k2//2) → ReLU``
    - ``AdaptiveMaxPool1d(T)`` → length T (default 8), so conv2 output is pooled to T bins
    - ``Flatten`` → ``Linear(C2*T → H) → ReLU → Dropout → Linear(H → 1)``

    ``in_channels`` must be 4 for the spacer one-hot path (see ``to_channel_first``).
    """

    def __init__(
        self,
        in_channels: int,
        *,
        conv1_channels: int = 114,
        conv2_channels: int = 124,
        kernel1: int = 3,
        kernel2: int = 3,
        pool_out: int = 16,
        head_hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if int(in_channels) != 4:
            raise ValueError(
                "CNN1DRegressorNet expects in_channels=4 (A,C,G,T one-hot). "
                f"Got in_channels={in_channels}."
            )
        p1 = kernel1 // 2
        p2 = kernel2 // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv1_channels, kernel_size=kernel1, padding=p1),
            nn.ReLU(),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel2, padding=p2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(pool_out),
        )
        flat = conv2_channels * pool_out
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, head_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x)).squeeze(-1)


def infer_cnn_in_channels(n_features: int) -> int:
    """DNA one-hot CNN always uses 4 input channels (A,C,G,T)."""
    assert_flat_dna_onehot_dim(n_features)
    return 4


def build_cnn_from_flat_dim(n_features: int, **kwargs: object) -> CNN1DRegressorNet:
    """Build CNN with ``in_channels=4`` after validating flat dim is L×4."""
    assert_flat_dna_onehot_dim(int(n_features))
    return CNN1DRegressorNet(4, **kwargs)
