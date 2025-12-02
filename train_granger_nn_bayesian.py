import torch
import torch.nn as nn
import numpy as np
from dataset.util import create_lagged_data, flatten_coordinates
from dataset.syn_data import generate_synthetic_protein_data
from model.granger_nn import GrangerNeuralNet
from model.util import coords_to_residue_scores, build_residue_adjacency, adjacency_to_digraph, plot_digraph
from tqdm.auto import trange


def compute_spatial_prior(data, threshold=10.0, heavy_penalty=10.0):
    T, N_res, C = data.shape

    mean_pos = np.mean(data, axis=0)

    diff = mean_pos[:, np.newaxis, :] - mean_pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    weights_res = np.ones((N_res, N_res), dtype=np.float32)
    weights_res[dist_matrix > threshold] = heavy_penalty

    np.fill_diagonal(weights_res, 1.0)

    prior_matrix = np.repeat(np.repeat(weights_res, C, axis=0), C, axis=1)

    return prior_matrix


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
    prior_matrix=None,
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

    if prior_matrix is not None:
        prior_matrix = torch.from_numpy(prior_matrix).float().to(device)

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

            if prior_matrix is not None:
                loss_sparse_v = (model.v.abs() * prior_matrix).sum()
            else:
                loss_sparse_v = model.v.abs().sum()

            loss_sparse_t = model.t.abs().sum()
            loss = loss_mse + lambda_v * loss_sparse_v + lambda_t * loss_sparse_t

            loss.backward()
            optimizer.step()

            epoch_loss += loss_mse.item()
            num_batches += 1

        avg_mse = epoch_loss / max(1, num_batches)

        current_l1_v = (lambda_v * loss_sparse_v).item()

        epoch_iter.set_postfix(
            mse=f"{avg_mse:.4f}",
            L1_v=f"{current_l1_v:.4f}",
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
    use_spatial_prior=True,
    prior_threshold=10.0,
    prior_penalty=10.0,
):
    T, D, C = data.shape

    prior_matrix = None
    if use_spatial_prior:
        print(
            f"Building spatial prior (Threshold={prior_threshold}A, Penalty={prior_penalty}x)..."
        )
        prior_matrix = compute_spatial_prior(data,
                                             threshold=prior_threshold,
                                             heavy_penalty=prior_penalty)

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
        prior_matrix=prior_matrix,  # 传入先验矩阵
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

    # 示例: 在 Positions 数据上开启 Spatial Prior
    print("\nTraining on Positions (with Spatial Prior)...")
    G_res_position, tau_position, S_res_position = train_granger_net(
        positions,
        use_spatial_prior=True,
        prior_threshold=8.0,
        prior_penalty=5.0)
    DG_res_position = adjacency_to_digraph(G_res_position,
                                           node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_position:.3f})"
    plot_digraph(DG_res_position, node_names=amino_acids, title=plot_title)

    # Angles 数据没有直观的空间距离概念(虽然也有，但更复杂)，这里演示不使用 Prior 或者你可以根据需要调整
    print("\nTraining on Angles (Standard)...")
    G_res_angle, tau_angle, S_res_angle = train_granger_net(
        angles,
        use_spatial_prior=False  # Angles 数据通常关闭 Spatial Prior，除非你有角度间的依赖先验
    )
    DG_res_angle = adjacency_to_digraph(G_res_angle, node_names=amino_acids)
    plot_title = f"Residue-level Granger Graph (tau = {tau_angle:.3f})"
    plot_digraph(DG_res_angle, node_names=amino_acids, title=plot_title)


if __name__ == "__main__":
    main()
