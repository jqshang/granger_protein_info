import numpy as np


def generate_synthetic_protein_data(num_acids, num_steps, pairs, seed=42):
    rng = np.random.default_rng(seed)

    # amino acid labels
    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]
    idx_map = {a: i for i, a in enumerate(amino_acids)}

    valid_pairs = [(Ai, Aj) for (Ai, Aj) in pairs
                   if Ai in idx_map and Aj in idx_map]

    # ---- Step 1: random initialization for all acids and times ----
    # positions: (T, N, 3), angles: (T, N, 2)
    positions = rng.uniform(-1.0, 1.0, size=(num_steps, num_acids, 3))
    angles = rng.uniform(-180.0, 180.0, size=(num_steps, num_acids, 2))

    # ---- Step 2: apply dependency rules over time ----
    for t in range(num_steps - 1):
        for Ai, Aj in valid_pairs:
            i = idx_map[Ai]
            j = idx_map[Aj]

            # copy previous values (overwrite any random at t+1)
            positions[t + 1, j, :] = positions[t, j, :]
            angles[t + 1, j, :] = angles[t, j, :]

            # apply update rule
            positions[t + 1,
                      j, :] += positions[t, j, :] * np.sin(positions[t, i, :])
            angles[t + 1, j, :] -= angles[t, i, :]

    return positions, angles, amino_acids
