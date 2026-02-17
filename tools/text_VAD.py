import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import os


vad_path = r""
vad_file = os.path.join(vad_path, "NRC-VAD-Lexicon-v2.1.txt")

data_path = r""
data_file = os.path.join(data_path, "transcriptions.csv")

# loading dict
vad_df = pd.read_csv(vad_file, sep="\t", names=["word", "valence", "arousal", "dominance"], skiprows=1)
vad_dict = vad_df.set_index("word").T.to_dict("list")  # {'happy': [0.89, 0.76, 0.67], ...}
# extract VAD
def extract_vad(text):
    tokens = word_tokenize(str(text).lower())
    vad_values = [vad_dict[word] for word in tokens if word in vad_dict]
    if not vad_values:
        return [0, 0, 0]  # 默认中性值
    avg = list(pd.DataFrame(vad_values).mean())
    return avg

# reading data
df = pd.read_csv(data_file)
valence_list = []
arousal_list = []
dominance_list = []

# main process
for text in df["TEXT"]:
    v, a, d = extract_vad(text)
    valence_list.append(v)
    arousal_list.append(a)
    dominance_list.append(d)

df["TEXT VALENCE"] = valence_list
df["TEXT AROUSAL"] = arousal_list
df["TEXT DOMINANCE"] = dominance_list

# save
output_file = os.path.join(data_path, "transcriptions_with_VAD.csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"VAD 提取完成！已保存到：\n{output_file}")
