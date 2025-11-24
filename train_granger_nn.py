import torch
import torch.nn as nn
from dataset.util import create_lagged_data, flatten_coordinates
from dataset.syn_data import generate_synthetic_protein_data
from model.granger_nn import GrangerNeuralNet
from model.util import coords_to_residue_scores, build_residue_adjacency, adjacency_to_digraph, plot_digraph


def _train_granger_net(X,
                       H=5,
                       hidden_dim=64,
                       num_layers=2,
                       lr=1e-3,
                       n_epochs=500,
                       lambda_v=1e-3,
                       lambda_t=1e-3,
                       device="cpu"):
    """
    X: numpy array of shape (T, D) = (time, features)
    Returns:
      model: trained GrangerNeuralNet
      S:     GC strength matrix (D, D)
    """
    X_torch = torch.from_numpy(X).float().to(device)  # (T, D)
    X_torch = X_torch.unsqueeze(0)  # (1, T, D)

    # standardize features over time
    with torch.no_grad():
        mean = X_torch.mean(dim=1, keepdim=True)
        std = X_torch.std(dim=1, keepdim=True) + 1e-6
        X_torch = (X_torch - mean) / std

    B, T, D = X_torch.shape
    X_lag, Y = create_lagged_data(X_torch, H)  # (N, H, D), (N, D)

    model = GrangerNeuralNet(D=D,
                             H=H,
                             hidden_dim=hidden_dim,
                             num_layers=num_layers,
                             device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        Y_hat = model(X_lag)  # (N, D)
        loss_mse = mse_loss(Y_hat, Y.to(device))

        # sparsity penalties
        loss_sparse_v = model.v.abs().sum()
        loss_sparse_t = model.t.abs().sum()
        loss = loss_mse + lambda_v * loss_sparse_v + lambda_t * loss_sparse_t

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"MSE: {loss_mse.item():.4f} | "
                  f"L1_v: {(lambda_v * loss_sparse_v).item():.4f} | "
                  f"L1_t: {(lambda_t * loss_sparse_t).item():.4f}")

    S = model.granger_matrix(p=2)
    return model, S


def train_granger_net(data):
    T, D, C = data.shape

    X = flatten_coordinates(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    H = 5  # number of lags
    model, S = _train_granger_net(
        X,
        H=H,
        hidden_dim=64,
        num_layers=2,
        lr=1e-3,
        n_epochs=300,
        lambda_v=1e-3,
        lambda_t=1e-3,
        device=device,
    )

    S_res = coords_to_residue_scores(S, n_coords=C)
    G_res, tau = build_residue_adjacency(S_res, percentile=95)

    return G_res, tau


def main():
    num_acids = 50
    num_steps = 200
    positions, angles, amino_acids = generate_synthetic_protein_data(
        num_acids, num_steps)

    G_res_position, tau_position = train_granger_net(positions)
    DG_res_position = adjacency_to_digraph(G_res_position,
                                           node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_position:.3f})"
    plot_digraph(DG_res_position, node_names=amino_acids, title=plot_title)

    G_res_angle, tau_angle = train_granger_net(angles)
    DG_res_angle = adjacency_to_digraph(G_res_angle, node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_angle:.3f})"
    plot_digraph(DG_res_angle, node_names=amino_acids, title=plot_title)
