import pandas as pd

# 输入和输出文件名（修改为你自己的路径）
input_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gpt_with_consistency_check.csv"         # 原始CSV文件路径
output_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gpr_inconsistent_only.csv"      # 输出文件路径

# 读取CSV
df = pd.read_csv(input_file)

# 筛选 consistency label 为 0 的样本
inconsistent_df = df[df["consistency label"] == 0]

# 保存到新CSV文件
inconsistent_df.to_csv(output_file, index=False)

print(f"已将 {len(inconsistent_df)} 个一致性为0的样本保存到：{output_file}")
