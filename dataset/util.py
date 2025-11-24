import numpy as np
import torch


def flatten_coordinates(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    T, N, C = data.shape

    # Flatten coordinates per residue: (T, N, 3) -> (T, N*3)
    X = data.reshape(T, N * C)
    return X


def create_lagged_data(x, H):
    """
    x: tensor of shape (B, T, D)
    H: number of lags

    Returns:
      X_lag: (N, H, D)
      Y:     (N, D)
    where N = B * (T - H)
    """
    B, T, D = x.shape
    if T <= H:
        raise ValueError(f"Need T > H; got T={T}, H={H}")

    lag_list = []
    y_list = []

    for b in range(B):
        for t in range(H, T):
            window = x[b, t - H:t, :]  # (H, D)
            target = x[b, t, :]  # (D,)
            lag_list.append(window.unsqueeze(0))
            y_list.append(target.unsqueeze(0))

    X_lag = torch.cat(lag_list, dim=0)  # (N, H, D)
    Y = torch.cat(y_list, dim=0)  # (N, D)
    return X_lag, Y
