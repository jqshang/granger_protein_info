import numpy as np


def syn_data(num_acids, num_steps, pairs, version="v1", seed=42):
    np.random.seed(seed)

    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]

    # ----- Step 1: random initialize everything -----
    pos = np.random.uniform(-1, 1, size=(num_steps, num_acids, 3))
    ang = np.random.uniform(-180, 180, size=(num_steps, num_acids, 2))

    # helper: map acid → index
    idx = {a: i for i, a in enumerate(amino_acids)}

    # ----- Step 2: apply chosen version -----
    for t in range(num_steps - 1):
        for Ai, Aj in pairs:
            i = idx[Ai]
            j = idx[Aj]

            Ai_pos = pos[t, i]
            Ai_ang = ang[t, i]
            Aj_pos_prev = pos[t, j]
            Aj_ang_prev = ang[t, j]

            if version == "v1":
                # copy previous
                pos[t + 1, j] = Aj_pos_prev.copy()
                ang[t + 1, j] = Aj_ang_prev.copy()
                # update
                pos[t + 1, j] += Aj_pos_prev * np.sin(Ai_pos)
                ang[t + 1, j] -= Ai_ang

            elif version == "v2":
                # angles copy only
                ang[t + 1, j] = Aj_ang_prev.copy()
                # update
                pos[t + 1, j] = Aj_pos_prev * np.sin(Ai_pos)
                ang[t + 1, j] -= Ai_ang

            elif version == "v3":
                # copy previous
                pos[t + 1, j] = Aj_pos_prev.copy()
                ang[t + 1, j] = Aj_ang_prev.copy()
                # update
                pos[t + 1, j] += np.sin(Ai_pos)
                ang[t + 1, j] -= Ai_ang

            else:
                raise ValueError("version must be 'v1', 'v2', or 'v3'")

    return pos, ang, amino_acids
