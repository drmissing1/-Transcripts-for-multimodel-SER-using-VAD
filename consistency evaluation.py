import pandas as pd

# 读取你的CSV文件
df = pd.read_csv("D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\merged_gpt_labels.csv")  # 请将文件名替换为你的实际文件名

# 判定是否一致（相等为1，否则为0）
df["一致性"] = (df["predicted_emotion"] == df["label"]).astype(int)

# 可选：计算一致率
consistency_rate = df["一致性"].mean() * 100
print(f"一致率为：{consistency_rate:.2f}%")

# 可选：保存结果为新CSV文件
df.to_csv("D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gpt_with_consistency_check.csv", index=False)
