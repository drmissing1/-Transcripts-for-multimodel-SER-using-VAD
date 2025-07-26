import pandas as pd
import numpy as np
from tqdm import tqdm  # 添加进度条

# === 步骤 1：读取数据文件 ===
file_path = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\enterface_vad_with_distance.csv"
df = pd.read_csv(file_path)

# === 步骤 2：提取欧氏距离列 ===
distances = df["EUCLIDEAN_DISTANCE"].values

# === 步骤 3：定义 Fisher Score 计算函数 ===
def fisher_score(X, y):
    labels = np.unique(y)
    mean_total = np.mean(X)
    numerator = 0
    denominator = 0
    for label in labels:
        x_class = X[y == label]
        n = len(x_class)
        mean_class = np.mean(x_class)
        var_class = np.var(x_class)
        numerator += n * (mean_class - mean_total) ** 2
        denominator += n * var_class
    return numerator / denominator if denominator > 0 else 0

# === 步骤 4：搜索最佳阈值 ===
min_d = np.min(distances)
max_d = np.max(distances)
step = 0.00001

best_threshold = None
best_fisher = -np.inf

# 使用 tqdm 包裹迭代器显示进度条
thresholds = np.arange(min_d, max_d, step)
for threshold in tqdm(thresholds, desc="Searching for best threshold"):
    labels = np.array(["consistent" if d <= threshold else "inconsistent" for d in distances])
    score = fisher_score(distances, labels)
    if score > best_fisher:
        best_fisher = score
        best_threshold = threshold

# === 步骤 5：输出结果 ===
print("✅ 步长搜索完成")
print(f"🎯 最优 Fisher Score 阈值: {best_threshold:.5f}")
print(f"📊 Fisher Score 得分: {best_fisher:.5f}")
