import pandas as pd
from scipy.stats import chi2_contingency

# === reading ===
df = pd.read_csv("D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\ds_with_consistency_check.csv")  # 请根据实际文件名替换

# === projecting ===
# consistent -> 0，inconsistent -> 1
df['modality_inconsistent'] = df['GMM_FINAL_LABEL'].map({'consistent': 0, 'inconsistent': 1})

df['LLM_wrong'] = 1 - df['consistency label']  

# === contingency tables ===
contingency_table = pd.crosstab(df['LLM_wrong'], df['modality_inconsistent'])

print(contingency_table)

# === chi-spuare test ===
chi2, p, dof, expected = chi2_contingency(contingency_table)

# print results
print("\n chi-square results:")
print(f"Chi-square value: {chi2:.4f}")
print(f"DOF: {dof}")
print(f"p value: {p:.10f}")
print("\n expected frequencies:")
print(expected)

# 根据 p 值判断显著性
alpha = 0.05
if p < alpha:
    print("\nConclusion: At the significance level of 0.05, there is a statistically significant relationship between modal inconsistency and LLM1 judgment errors.")
else:
    print("\nConclusion: At the significance level of 0.05, there is no statistically significant relationship between modal inconsistency and LLM1 judgment errors.")
