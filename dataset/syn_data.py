import numpy as np


def generate_synthetic_protein_data_v3_fixed(num_acids,
                                             num_steps,
                                             pairs,
                                             seed=42):
    np.random.seed(seed)
    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]
    idx = {a: i for i, a in enumerate(amino_acids)}

    # 初始化：全零矩阵，只随机初始化 t=0
    pos = np.zeros((num_steps, num_acids, 3))
    ang = np.zeros((num_steps, num_acids, 2))

    pos[0] = np.random.uniform(-1, 1, size=(num_acids, 3))
    ang[0] = np.random.uniform(-180, 180, size=(num_acids, 2))

    # 耦合强度系数 (防止数值发散)
    coupling_strength = 0.1
    noise_level = 0.01

    for t in range(num_steps - 1):
        # step 1: 先让所有人根据自身上一时刻状态进行更新 (自我回归)
        # 这一步至关重要，保证了数据有“记忆性”，而不是白噪声
        pos[t +
            1] = pos[t] + np.random.normal(0, noise_level, size=(num_acids, 3))
        ang[t +
            1] = ang[t] + np.random.normal(0, noise_level, size=(num_acids, 2))

        # step 2: 遍历 pairs，叠加因果效应 (使用 +=)
        for Ai, Aj in pairs:
            i = idx[Ai]
            j = idx[Aj]

            # 获取源节点在 t 时刻的状态
            Ai_pos = pos[t, i]
            Ai_ang = ang[t, i]

            # 【关键修改】直接在 t+1 已经存在的值上 *累加* 效应
            # 你的 v3 逻辑是 sin(source)
            pos[t + 1, j] += np.sin(Ai_pos) * coupling_strength
            ang[t + 1, j] -= Ai_ang * coupling_strength

        # step 3: 统一处理角度归一化 (放在循环外或所有update之后)
        ang[t + 1] = ((ang[t + 1] + 180) % 360) - 180

    return pos, ang, amino_acids
