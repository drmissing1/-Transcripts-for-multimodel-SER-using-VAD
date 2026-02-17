import pandas as pd
import numpy as np
from tqdm import tqdm 

# === reading ===
file_path = ""
df = pd.read_csv(file_path)
distances = df["EUCLIDEAN_DISTANCE"].values

# === def fisher score ===
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

# === search for best threshold ===
min_d = np.min(distances)
max_d = np.max(distances)
step = 0.00001

best_threshold = None
best_fisher = -np.inf
# === loading schedule ===
thresholds = np.arange(min_d, max_d, step)
for threshold in tqdm(thresholds, desc="Searching for best threshold"):
    labels = np.array(["consistent" if d <= threshold else "inconsistent" for d in distances])
    score = fisher_score(distances, labels)
    if score > best_fisher:
        best_fisher = score
        best_threshold = threshold

# === print ===
print(f"optimal Fisher Score threshold: {best_threshold:.5f}")
print(f"Fisher Score: {best_fisher:.5f}")
