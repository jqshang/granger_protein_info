import numpy as np
import pandas as pd


def generate_synthetic_protein_data(num_acids, num_steps, seed=42):
    np.random.seed(seed)
    """Generates synthetic protein folding data with dependencies among amino acids."""
    # List of amino acid identifiers
    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]

    # dependency pairs Ai -> Aj
    pairs = [("A1", "A16"), ("A16", "A17"), ("A17", "A30"), ("A30", "A36"),
             ("A30", "A16"), ("A36", "A50")]

    # amino acids affected by rules (targets only)
    dependent_targets = {"A16", "A17", "A30", "A36", "A50"}

    # ----- Step 1: Generate full random dataset for all acids and all times -----
    data = pd.DataFrame(index=range(num_steps), columns=amino_acids)

    for t in range(num_steps):
        for a in amino_acids:
            pos = np.random.uniform(-1, 1, size=3)
            angles = np.random.uniform(-180, 180, size=2)
            data.at[t, a] = {"pos": pos, "angles": angles}

    # ----- Step 2: Overwrite dependent amino acids using the rules -----
    for t in range(num_steps - 1):
        for Ai, Aj in pairs:
            Ai_prev = data.at[t, Ai]
            Aj_prev = data.at[t, Aj]
            Aj_next = data.at[t + 1, Aj]

            # overwrite only Aj (the target)
            # step 1: copy previous values
            Aj_next["pos"] = Aj_prev["pos"].copy()
            Aj_next["angles"] = Aj_prev["angles"].copy()

            # step 2: apply update rule
            Aj_next["pos"] += Aj_prev["pos"] * np.sin(Ai_prev["pos"])
            Aj_next["angles"] -= Ai_prev["angles"]

    return data, amino_acids, dependent_targets, pairs
