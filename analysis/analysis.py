import pandas as pd

# === reading ===
gmm_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gmm_classification_output.csv"
inconsistent_file = "D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\gpt_inconsistent_only.csv"

gmm_df = pd.read_csv(gmm_file)
inconsistent_df = pd.read_csv(inconsistent_file)

# === confirm ID ===
gmm_id_col = 'number' if 'number' in gmm_df.columns else gmm_df.columns[0]
inconsistent_id_col = 'number' if 'number' in inconsistent_df.columns else inconsistent_df.columns[0]

gmm_df[gmm_id_col] = gmm_df[gmm_id_col].astype(str)
inconsistent_df[inconsistent_id_col] = inconsistent_df[inconsistent_id_col].astype(str)
error_ids = inconsistent_df[inconsistent_id_col].tolist()

# === matching samples ===
matching_samples = gmm_df[gmm_df[gmm_id_col].isin(error_ids)]   

# === select incongruent results ===
inconsistent_count = (matching_samples["GMM_FINAL_LABEL"] == "consistent").sum()
total_count = len(error_ids)

# === calculate ===
percentage = (inconsistent_count / total_count) * 100 if total_count > 0 else 0

print(f"general inconguent samples: {total_count}")
print(f"inconsistent samples: {inconsistent_count}")
print(f"ratio: {percentage:.2f}%")
