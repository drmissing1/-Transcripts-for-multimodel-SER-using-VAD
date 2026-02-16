import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ========== 🧠 设置完全确定性 ==========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ========== 📁 路径设置 ==========
wav_dir = r"D:\Multimodel SER system\project2_database\enterface wave"
csv_path = r"D:\Multimodel SER system\project2_database\outputs\transcriptions_with_VAD.csv"

# ========== 🤖 模型加载 ==========
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
model.eval()
device = torch.device("cpu")  # 如果想用GPU，可改为 "cuda"
model.to(device)

# ========== 🧾 CSV读取 ==========
df = pd.read_csv(csv_path)
valences, arousals, dominances = [], [], []

# ========== 🔁 音频遍历 ==========
for i, row in df.iterrows():
    wav_path = os.path.join(wav_dir, row["REC.WAV"])
    if not os.path.exists(wav_path):
        print(f"⚠️ 文件未找到：{wav_path}")
        valences.append(0.0)
        arousals.append(0.0)
        dominances.append(0.0)
        continue

    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.clone().detach()  # 防止非确定性修改

    # 单声道化
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # 转为 torch tensor（不再用 numpy）
    input_tensor = waveform.squeeze(0).to(device)
    inputs = extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()

    valences.append(float(logits[0]))
    arousals.append(float(logits[1]))
    dominances.append(float(logits[2]))

# ========== 💾 写入新CSV ==========
df["AUDIO VALENCE"] = valences
df["AUDIO AROUSAL"] = arousals
df["AUDIO DOMINANCE"] = dominances

output_csv = csv_path.replace(".csv", "_with_AUDIO_VAD_deterministic1.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ 音频情绪分析完成，共处理 {len(df)} 条音频。\n保存结果至：{output_csv}")
