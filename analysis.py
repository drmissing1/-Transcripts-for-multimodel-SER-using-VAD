import pandas as pd

# === 步骤 1：读取数据文件 ===
gmm_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gmm_classification_output.csv"
inconsistent_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gpt_inconsistent_only.csv"

gmm_df = pd.read_csv(gmm_file)
inconsistent_df = pd.read_csv(inconsistent_file)

# === 步骤 2：确定ID列名（默认使用首列或名为"ID"的列）===
gmm_id_col = 'number' if 'number' in gmm_df.columns else gmm_df.columns[0]
inconsistent_id_col = 'number' if 'number' in inconsistent_df.columns else inconsistent_df.columns[0]

# === 步骤 3：统一ID类型为字符串，获取错误样本ID列表 ===
gmm_df[gmm_id_col] = gmm_df[gmm_id_col].astype(str)
inconsistent_df[inconsistent_id_col] = inconsistent_df[inconsistent_id_col].astype(str)
error_ids = inconsistent_df[inconsistent_id_col].tolist()

# === 步骤 4：在GMM数据中查找匹配样本 ===
matching_samples = gmm_df[gmm_df[gmm_id_col].isin(error_ids)]   

# === 步骤 5：统计其中被标为 inconsistent 的样本数量 ===
inconsistent_count = (matching_samples["GMM_FINAL_LABEL"] == "consistent").sum()
total_count = len(error_ids)

# === 步骤 6：计算百分比并输出结果 ===
percentage = (inconsistent_count / total_count) * 100 if total_count > 0 else 0

print(f"🎯 总 LLM 出错样本数: {total_count}")
print(f"📊 其中 GMM 判定为 inconsistent 的数量: {inconsistent_count}")
print(f"📈 占比: {percentage:.2f}%")
