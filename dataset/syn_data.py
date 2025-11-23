import numpy as np


def generate_synthetic_protein_data(num_acids, num_steps, seed=42):
    np.random.seed(seed)

    # List of amino acid identifiers
    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]

    # dependency pairs Ai -> Aj
    pairs = [("A1", "A16"), ("A16", "A17"), ("A17", "A30"), ("A30", "A36"),
             ("A30", "A16"), ("A36", "A50")]

    # amino acids affected by rules (targets only)
    dependent_targets = {"A16", "A17", "A30", "A36", "A50"}

    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    pos = np.random.uniform(-1, 1, size=(num_steps, num_acids, 3))
    angles = np.random.uniform(-180, 180, size=(num_steps, num_acids, 2))

    for t in range(num_steps - 1):
        for Ai, Aj in pairs:
            i = aa_to_idx[Ai]
            j = aa_to_idx[Aj]

            Ai_prev_pos = pos[t, i]
            Ai_prev_angles = angles[t, i]
            Aj_prev_pos = pos[t, j]
            Aj_prev_angles = angles[t, j]

            pos[t + 1, j] = Aj_prev_pos + Aj_prev_pos * np.sin(Ai_prev_pos)
            angles[t + 1, j] = Aj_prev_angles - Ai_prev_angles

    pos_array = pos.reshape(num_steps, num_acids * 3)
    angle_array = angles.reshape(num_steps, num_acids * 2)

    return pos_array, angle_array, amino_acids, dependent_targets, pairs
