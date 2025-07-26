import pandas as pd
from scipy.stats import chi2_contingency

# 读取数据文件
df = pd.read_csv("D:\\Multimodel SER system\\Multimodel SER System 1\\outputs\\ds_with_consistency_check.csv")  # 请根据实际文件名替换

# 映射字符串标签为数值（方便处理）
# 一致 -> 0，不一致 -> 1
df['modality_inconsistent'] = df['GMM_FINAL_LABEL'].map({'consistent': 0, 'inconsistent': 1})

# consistency label 已经是 0（错误）和 1（正确），我们构造 LLM 是否错误分类标签
df['LLM_wrong'] = 1 - df['consistency label']  # 转换为 1 = 错误，0 = 正确

# 构造列联表
contingency_table = pd.crosstab(df['LLM_wrong'], df['modality_inconsistent'])

# 打印列联表（便于检查）
print("列联表（LLM是否出错 vs 模态是否不一致）:")
print(contingency_table)

# 执行卡方检验
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 输出结果
print("\n卡方检验结果:")
print(f"Chi-square 统计量: {chi2:.4f}")
print(f"自由度: {dof}")
print(f"p 值: {p:.10f}")
print("\n期望频数 (expected frequencies):")
print(expected)

# 根据 p 值判断显著性
alpha = 0.05
if p < alpha:
    print("\n结论：在显著性水平 0.05 下，模态不一致 与 LLM1 判断错误 存在统计显著关系。")
else:
    print("\n结论：在显著性水平 0.05 下，未发现模态不一致 与 LLM1 判断错误 有显著统计关系。")
