import torch
import torch.nn as nn
from dataset.util import create_lagged_data, flatten_coordinates
from dataset.syn_data import generate_synthetic_protein_data
from model.granger_nn import GrangerNeuralNet
from model.util import coords_to_residue_scores, build_residue_adjacency, adjacency_to_digraph, plot_digraph
from tqdm.auto import trange


def _train_granger_net(
    X,
    H=5,
    hidden_dim=64,
    num_layers=2,
    lr=1e-3,
    n_epochs=500,
    lambda_v=1e-4,
    lambda_t=1e-4,
    device="cpu",
    batch_size=256,
):
    X_torch = torch.from_numpy(X).float()  # (T, D) on CPU
    X_torch = X_torch.unsqueeze(0)  # (1, T, D)

    with torch.no_grad():
        mean = X_torch.mean(dim=1, keepdim=True)
        std = X_torch.std(dim=1, keepdim=True) + 1e-6
        X_torch = (X_torch - mean) / std

    B, T, D = X_torch.shape
    X_lag, Y = create_lagged_data(X_torch, H)  # (N, H, D), (N, D)
    N = X_lag.shape[0]

    model = GrangerNeuralNet(
        D=D,
        H=H,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    epoch_iter = trange(n_epochs, desc="Training Granger NN", leave=True)

    for epoch in epoch_iter:
        model.train()

        perm = torch.randperm(N)

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]

            X_batch = X_lag[idx].to(device)
            Y_batch = Y[idx].to(device)

            optimizer.zero_grad(set_to_none=True)

            Y_hat = model(X_batch)  # (batch, D)
            loss_mse = mse_loss(Y_hat, Y_batch)

            loss_sparse_v = model.v.abs().sum()
            loss_sparse_t = model.t.abs().sum()
            loss = loss_mse + lambda_v * loss_sparse_v + lambda_t * loss_sparse_t

            loss.backward()
            optimizer.step()

            epoch_loss += loss_mse.item()
            num_batches += 1

        avg_mse = epoch_loss / max(1, num_batches)
        epoch_iter.set_postfix(
            mse=f"{avg_mse:.4f}",
            L1_v=f"{(lambda_v * model.v.abs().sum()).item():.4f}",
            L1_t=f"{(lambda_t * model.t.abs().sum()).item():.4f}",
        )

    S = model.granger_matrix(p=2)
    return model, S


def train_granger_net(
    data,
    H=5,
    hidden_dim=64,
    num_layers=2,
    lr=1e-3,
    n_epochs=500,
    lambda_v=1e-3,
    lambda_t=1e-3,
    batch_size=256,
):
    T, D, C = data.shape

    X = flatten_coordinates(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, S = _train_granger_net(
        X,
        H=H,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        n_epochs=n_epochs,
        lambda_v=lambda_v,
        lambda_t=lambda_t,
        device=device,
        batch_size=batch_size,
    )

    S_res = coords_to_residue_scores(S, n_coords=C)
    G_res, tau = build_residue_adjacency(S_res, percentile=95)

    return G_res, tau, S_res


def main():
    num_acids = 50
    num_steps = 200
    pairs = [("A1", "A16"), ("A16", "A17"), ("A17", "A30"), ("A30", "A36"),
             ("A30", "A16"), ("A36", "A50")]
    positions, angles, amino_acids = generate_synthetic_protein_data(
        num_acids, num_steps, pairs)

    G_res_position, tau_position, S_res_position = train_granger_net(positions)
    DG_res_position = adjacency_to_digraph(G_res_position,
                                           node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_position:.3f})"
    plot_digraph(DG_res_position, node_names=amino_acids, title=plot_title)

    G_res_angle, tau_angle, S_res_angle = train_granger_net(angles)
    DG_res_angle = adjacency_to_digraph(G_res_angle, node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_angle:.3f})"
    plot_digraph(DG_res_angle, node_names=amino_acids, title=plot_title)
