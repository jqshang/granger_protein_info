import numpy as np


def generate_synthetic_protein_data(num_acids, num_steps, pairs, seed=42):
    np.random.seed(seed)

    # ... 初始化代码不变 ...
    pos = np.zeros((num_steps, num_acids, 3))
    ang = np.zeros((num_steps, num_acids, 2))

    pos[0] = np.random.uniform(-1, 1, size=(num_acids, 3))
    ang[0] = np.random.uniform(-180, 180, size=(num_acids, 2))

    coupling_strength = 0.2  # 稍微加大一点信号
    noise_level = 0.05  # 必须保留噪声，但可以小一点

    # 【核心修改】：衰减系数 (Decay/Friction)
    # 0.9 表示每一步位置都会衰减 10%，这就产生了一个“拉回原点”的力
    decay = 0.9

    for t in range(num_steps - 1):
        # Step 1: 自回归 + 均值回归 + 噪声
        # 即使没有因果输入，原子也会在一个范围内震荡，而不会漂移到无穷远
        pos[t + 1] = pos[t] * decay + np.random.normal(
            0, noise_level, size=(num_acids, 3))
        ang[t + 1] = ang[t] * decay + np.random.normal(
            0, noise_level, size=(num_acids, 2))

        # Step 2: 因果叠加 (保持不变)
        for Ai, Aj in pairs:
            i, j = idx[Ai], idx[Aj]

            # 使用 += 累加效应
            # 注意：这里加上 sin 信号后，由于有 decay，信号也会随时间消散，这是符合物理的
            pos[t + 1, j] += np.sin(pos[t, i]) * coupling_strength
            ang[t + 1, j] -= ang[t, i] * coupling_strength

        # Step 3: 角度归一化
        ang[t + 1] = ((ang[t + 1] + 180) % 360) - 180

    return pos, ang, amino_acids
