import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 加载数据
data = pd.read_csv('D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\transcriptions_with_audio_VAD.csv', encoding='utf-8-sig')

# print(data.columns.tolist())

# 计算欧氏距离
def euclidean_distance(row):
    text_vad = np.array([row['TEXT VALENCE'], row['TEXT AROUSAL'], row['TEXT DOMINANCE']])
    audio_vad = np.array([row['AUDIO VALENCE'], row['AUDIO AROUSAL'], row['AUDIO DOMINANCE']])
    return np.linalg.norm(text_vad - audio_vad)

# 创建欧式距离新列
data['EUCLIDEAN_DISTANCE'] = data.apply(euclidean_distance, axis=1)

# # 设置阈值（可随时更改）
# threshold = 0.3

# # 根据阈值创建二元标签
# data['CONSISTENCY_LABEL'] = data['EUCLIDEAN_DISTANCE'].apply(lambda x: 'consistent' if x <= threshold else 'inconsistent')

# Fisher Score函数定义
# def fisher_score(feature, label):
#     labels_unique = np.unique(label)
#     numerator = sum(len(feature[label == l]) * (np.mean(feature[label == l]) - np.mean(feature))**2 for l in labels_unique)
#     denominator = sum(len(feature[label == l]) * np.var(feature[label == l]) for l in labels_unique)
#     return numerator / denominator if denominator != 0 else np.nan

# 计算三个指标
# features = data[['TEXT VALENCE', 'TEXT AROUSAL', 'TEXT DOMINANCE', 'AUDIO VALENCE', 'AUDIO AROUSAL', 'AUDIO DOMINANCE']].values
# labels_binary = data['CONSISTENCY_LABEL'].apply(lambda x: 1 if x == 'consistent' else 0).values

# fisher = fisher_score(data['EUCLIDEAN_DISTANCE'].values, labels_binary)
# silhouette = silhouette_score(features, labels_binary)
# db_index = davies_bouldin_score(features, labels_binary)

# 将指标添加到新列（每行相同指标值）
# data['FISHER_SCORE'] = fisher
# data['SILHOUETTE_COEFF'] = silhouette
# data['DAVIES_BOULDIN_INDEX'] = db_index

# 保存新数据文件
data.to_csv('D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\enterface_vad_with_distance.csv', index=False, encoding='utf-8-sig')

print("计算完成并已保存！")
