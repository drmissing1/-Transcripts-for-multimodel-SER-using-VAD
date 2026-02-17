import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

# === loading data ===
data = pd.read_csv('', encoding='utf-8-sig')


# === calculate distance ===
def euclidean_distance(row):
    text_vad = np.array([row['TEXT VALENCE'], row['TEXT AROUSAL'], row['TEXT DOMINANCE']])
    audio_vad = np.array([row['AUDIO VALENCE'], row['AUDIO AROUSAL'], row['AUDIO DOMINANCE']])
    return np.linalg.norm(text_vad - audio_vad)

data['EUCLIDEAN_DISTANCE'] = data.apply(euclidean_distance, axis=1)

# === set threshold ===
threshold = 0.3

# === create labels ===
data['CONSISTENCY_LABEL'] = data['EUCLIDEAN_DISTANCE'].apply(lambda x: 'consistent' if x <= threshold else 'inconsistent')

def fisher_score(feature, label):
    labels_unique = np.unique(label)
    numerator = sum(len(feature[label == l]) * (np.mean(feature[label == l]) - np.mean(feature))**2 for l in labels_unique)
    denominator = sum(len(feature[label == l]) * np.var(feature[label == l]) for l in labels_unique)
    return numerator / denominator if denominator != 0 else np.nan

features = data[['TEXT VALENCE', 'TEXT AROUSAL', 'TEXT DOMINANCE', 'AUDIO VALENCE', 'AUDIO AROUSAL', 'AUDIO DOMINANCE']].values
labels_binary = data['CONSISTENCY_LABEL'].apply(lambda x: 1 if x == 'consistent' else 0).values

fisher = fisher_score(data['EUCLIDEAN_DISTANCE'].values, labels_binary)
silhouette = silhouette_score(features, labels_binary)
db_index = davies_bouldin_score(features, labels_binary)

data['FISHER_SCORE'] = fisher
data['SILHOUETTE_COEFF'] = silhouette
data['DAVIES_BOULDIN_INDEX'] = db_index

# === save ===
data.to_csv('', index=False, encoding='utf-8-sig')

