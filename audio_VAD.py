import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from tqdm import tqdm

# ==== 定义模型结构 ====
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

# ==== 设置路径 ====
csv_path = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\transcriptions_with_text_VAD.csv"
audio_dir = "D:\\Multimodel SER system\\project2_database\\enterface wave"
output_path = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\transcriptions_with_audio_VAD.csv"

# ==== 加载模型 ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)
model.eval()

# ==== 推理函数 ====
def extract_vad(signal: np.ndarray, sr: int = 16000):
    y = processor(signal, sampling_rate=sr)
    input_tensor = torch.tensor(y["input_values"]).to(device)
    with torch.no_grad():
        _, logits = model(input_tensor)
    vad = logits.squeeze().cpu().numpy()  # 原始 [arousal, dominance, valence]

    vad = np.clip(vad, 0.0, 1.0)

    # ✅ 映射到 [-1, 1] 区间
    vad = (vad - 0.5) * 2

    return vad  # 仍然顺序是 [arousal, dominance, valence]

# ==== 读取CSV ====
df = pd.read_csv(csv_path)
vad_results = []

# ==== 遍历每一行处理音频 ====
for i, row in tqdm(df.iterrows(), total=len(df)):
    wav_name = row["REC.WAV"]
    wav_path = os.path.join(audio_dir, wav_name)

    if not os.path.exists(wav_path):
        print(f"⚠️ 未找到音频文件: {wav_path}")
        vad_results.append([None, None, None])
        continue

    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    signal = waveform.squeeze(0).numpy()

    vad = extract_vad(signal, 16000)
    valence = float(vad[2])
    arousal = float(vad[0])
    dominance = float(vad[1])
    vad_results.append([valence, arousal, dominance])

# ==== 写入新列 ====
df["AUDIO VALENCE"] = [v[0] for v in vad_results]
df["AUDIO AROUSAL"] = [v[1] for v in vad_results]
df["AUDIO DOMINANCE"] = [v[2] for v in vad_results]

df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 音频处理完毕，已保存到：{output_path}")
