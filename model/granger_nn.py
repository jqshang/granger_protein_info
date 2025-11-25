import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers):
            lin = nn.Linear(d_in, hidden_dim)
            lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            layers.append(nn.ReLU())
            d_in = hidden_dim
        out = nn.Linear(d_in, 1)
        out = nn.utils.weight_norm(out)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, input_dim)
        return self.net(x).squeeze(-1)  # (N,)


class GrangerNeuralNet(nn.Module):
    """
    Decoupled Granger Neural Net:
      - v[d, j]  : importance of source series j for target d
      - t[d, h, j]: importance of lag h of source j for target d
      - one MLP per target dimension d
    """

    def __init__(self, D, H, hidden_dim=128, num_layers=4, device="cpu"):
        super().__init__()
        self.D = D
        self.H = H
        self.device = device

        # v[d, j]: (D, D)
        self.v = nn.Parameter(torch.ones(D, D))

        # t[d, h, j]: (D, H, D)
        self.t = nn.Parameter(torch.ones(D, H, D))

        # one small MLP per target dimension
        self.mlps = nn.ModuleList([
            MLP(H * D, hidden_dim=hidden_dim, num_layers=num_layers)
            for _ in range(D)
        ])

        self.to(device)

    def forward(self, X_lag):
        """
        X_lag: (N, H, D)
        Returns:
          Y_hat: (N, D)
        """
        N, H, D = X_lag.shape
        assert H == self.H and D == self.D

        X_lag = X_lag.to(self.device)
        Y_hat_list = []

        for d in range(self.D):
            v_d = self.v[d]  # (D,)
            t_d = self.t[d]  # (H, D)

            # broadcast to (N, H, D)
            x_scaled = X_lag * v_d.view(1, 1, D) * t_d.view(1, H, D)
            x_flat = x_scaled.view(N, H * D)

            y_hat_d = self.mlps[d](x_flat)  # (N,)
            Y_hat_list.append(y_hat_d.unsqueeze(-1))

        Y_hat = torch.cat(Y_hat_list, dim=1)  # (N, D)
        return Y_hat

    def granger_matrix(self, p=2):
        """
        Extract GC strengths using group norm over lags:
          S[d, j] = |v[d, j]| * ||t[d, :, j]||_p
        Returns:
          S: (D, D) numpy array
        """
        with torch.no_grad():
            v = self.v
            t = self.t

            if p == 1:
                t_norm = t.abs().sum(dim=1)  # (D, D)
            else:
                t_norm = torch.sqrt((t**2).sum(dim=1) + 1e-12)  # (D, D)

            S = v.abs() * t_norm
            return S.detach().cpu().numpy()
