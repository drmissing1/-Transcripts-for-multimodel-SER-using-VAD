import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import fsolve
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

# === 步骤 1：加载数据 ===
file_path = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\enterface_vad_with_distance.csv"
df = pd.read_csv(file_path)

X = df["EUCLIDEAN_DISTANCE"].values.reshape(-1, 1)

# === 步骤 2：拟合两个成分的 GMM ===
gmm = GaussianMixture(n_components=2, init_params='kmeans', random_state=42)
gmm.fit(X)

# === 步骤 3：参数排序并求交点 ===
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_

sorted_indices = np.argsort(means)
means = means[sorted_indices]
stds = stds[sorted_indices]
weights = weights[sorted_indices]

def gmm_pdf(x, weight, mean, std):
    return weight * norm.pdf(x, mean, std)

def diff(x):
    return gmm_pdf(x, weights[0], means[0], stds[0]) - gmm_pdf(x, weights[1], means[1], stds[1])

threshold = fsolve(diff, np.mean(means))[0]

# === 步骤 4：输出软概率和最终一致性标签 ===
probabilities = gmm.predict_proba(X)
df['GMM_PROB_0'] = probabilities[:, sorted_indices[0]]
df['GMM_PROB_1'] = probabilities[:, sorted_indices[1]]
df['GMM_COMPONENT'] = np.argmax(probabilities[:, sorted_indices], axis=1)
df['GMM_FINAL_LABEL'] = df['EUCLIDEAN_DISTANCE'].apply(lambda x: 'consistent' if x < threshold else 'inconsistent')
df['GMM_THRESHOLD'] = threshold

# === 步骤 5：计算三个指标 ===

# Fisher Score（简单定义：类间方差 / 类内方差）
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

# 需要将标签转换为整数
le = LabelEncoder()
cluster_labels = le.fit_transform(df['GMM_FINAL_LABEL'])

fisher = fisher_score(X.flatten(), df['GMM_FINAL_LABEL'].values)
silhouette = silhouette_score(X, cluster_labels)
dbi = davies_bouldin_score(X, cluster_labels)

# === 步骤 6：保存文件 + 输出指标 ===
output_path = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gmm_classification_output1.csv"
df.to_csv(output_path, index=False)

print("✅ 分析完成，结果文件已保存为:", output_path)
print(f"🟨 贝叶斯最优阈值为: {threshold:.5f}")
print(f"📊 Fisher Score: {fisher:.5f}")
print(f"📊 Silhouette Coefficient: {silhouette:.5f}")
print(f"📊 Davies-Bouldin Index: {dbi:.5f}")
