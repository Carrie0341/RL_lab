import numpy as np
import matplotlib.pyplot as plt
import torch

# 定義squashing函數
def squashing_fn(x, a, b):
    return 1 / (np.exp(a * x) + b + np.exp(-a * x))

# 設定參數
keypoint_coef_baseline = [5, 4]  # General movement towards fixed object
keypoint_coef_coarse = [50, 2]   # Movement to align the assets
keypoint_coef_fine = [100, 0]    # Smaller distances for threading or last-inch insertion

# 創建keypoint_dist的範圍從0到0.1
keypoint_dist = np.linspace(0, 0.5, 1000)

# 計算每個獎勵函數的值
kp_baseline = squashing_fn(keypoint_dist, keypoint_coef_baseline[0], keypoint_coef_baseline[1])
kp_coarse = squashing_fn(keypoint_dist, keypoint_coef_coarse[0], keypoint_coef_coarse[1])
kp_fine = squashing_fn(keypoint_dist, keypoint_coef_fine[0], keypoint_coef_fine[1])

# 計算總獎勵 (不包括動作懲罰)
total_reward = kp_baseline + kp_coarse + kp_fine

# 創建圖表
plt.figure(figsize=(12, 8))

# 繪製每個獎勵函數
plt.plot(keypoint_dist, kp_baseline, label=f'kp_baseline (a={keypoint_coef_baseline[0]}, b={keypoint_coef_baseline[1]})', linewidth=2)
plt.plot(keypoint_dist, kp_coarse, label=f'kp_coarse (a={keypoint_coef_coarse[0]}, b={keypoint_coef_coarse[1]})', linewidth=2)
plt.plot(keypoint_dist, kp_fine, label=f'kp_fine (a={keypoint_coef_fine[0]}, b={keypoint_coef_fine[1]})', linewidth=2)
plt.plot(keypoint_dist, total_reward, label='Total (baseline + coarse + fine)', linewidth=3, linestyle='--')

# 添加網格線
plt.grid(True, alpha=0.3)

# 設置軸標籤和標題
plt.xlabel('Keypoint Distance', fontsize=14)
plt.ylabel('Reward Value', fontsize=14)
plt.title('Reward Components vs Keypoint Distance', fontsize=16)

# 添加圖例
plt.legend(fontsize=12)

# 設置y軸範圍以更好地顯示曲線
plt.ylim(0, 3.0)

# 添加垂直線來標記重要距離
plt.axvline(x=0.01, color='gray', linestyle=':', alpha=0.7, label='Distance=0.01')
plt.axvline(x=0.02, color='gray', linestyle=':', alpha=0.7, label='Distance=0.02')
plt.axvline(x=0.05, color='gray', linestyle=':', alpha=0.7, label='Distance=0.05')

# 添加詳細的分析表格
distances = np.linspace(0, 0.5, 10)
table_data = []

for dist in distances:
    baseline = squashing_fn(dist, keypoint_coef_baseline[0], keypoint_coef_baseline[1])
    coarse = squashing_fn(dist, keypoint_coef_coarse[0], keypoint_coef_coarse[1])
    fine = squashing_fn(dist, keypoint_coef_fine[0], keypoint_coef_fine[1])
    total = baseline + coarse + fine
    table_data.append([dist, baseline, coarse, fine, total])

# 創建表格來顯示在特定距離的獎勵值
table_ax = plt.table(
    cellText=[[f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"] for row in table_data],
    colLabels=["Distance", "Baseline", "Coarse", "Fine", "Total"],
    loc='bottom',
    bbox=[0.0, -0.35, 1.0, 0.25]
)
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(10)
table_ax.scale(1, 1.5)

# 調整圖表布局以適應表格
plt.subplots_adjust(bottom=0.3)

# 保存圖表
plt.savefig('reward_visualization.png', dpi=300, bbox_inches='tight')

# 顯示圖表
plt.show()

# # 創建第二個圖表，專注於小距離範圍
# plt.figure(figsize=(12, 8))

# # 創建更小範圍的keypoint_dist
# small_range_dist = np.linspace(0, 0.02, 1000)

# # 計算每個獎勵函數的值
# small_kp_baseline = squashing_fn(small_range_dist, keypoint_coef_baseline[0], keypoint_coef_baseline[1])
# small_kp_coarse = squashing_fn(small_range_dist, keypoint_coef_coarse[0], keypoint_coef_coarse[1])
# small_kp_fine = squashing_fn(small_range_dist, keypoint_coef_fine[0], keypoint_coef_fine[1])
# small_total_reward = small_kp_baseline + small_kp_coarse + small_kp_fine

# # 繪製每個獎勵函數
# plt.plot(small_range_dist, small_kp_baseline, label=f'kp_baseline (a={keypoint_coef_baseline[0]}, b={keypoint_coef_baseline[1]})', linewidth=2)
# plt.plot(small_range_dist, small_kp_coarse, label=f'kp_coarse (a={keypoint_coef_coarse[0]}, b={keypoint_coef_coarse[1]})', linewidth=2)
# plt.plot(small_range_dist, small_kp_fine, label=f'kp_fine (a={keypoint_coef_fine[0]}, b={keypoint_coef_fine[1]})', linewidth=2)
# plt.plot(small_range_dist, small_total_reward, label='Total (baseline + coarse + fine)', linewidth=3, linestyle='--')

# # 添加網格線
# plt.grid(True, alpha=0.3)

# # 設置軸標籤和標題
# plt.xlabel('Keypoint Distance (Small Range)', fontsize=14)
# plt.ylabel('Reward Value', fontsize=14)
# plt.title('Reward Components vs Keypoint Distance (0-0.02)', fontsize=16)

# # 添加圖例
# plt.legend(fontsize=12)

# # 設置y軸範圍
# plt.ylim(0, 3.0)

# # 添加垂直線來標記重要距離
# plt.axvline(x=0.001, color='gray', linestyle=':', alpha=0.7)
# plt.axvline(x=0.005, color='gray', linestyle=':', alpha=0.7)
# plt.axvline(x=0.01, color='gray', linestyle=':', alpha=0.7)

# # 保存小範圍圖表
# plt.savefig('reward_visualization_small_range.png', dpi=300, bbox_inches='tight')

# # 顯示圖表
# plt.show()

# # 打印重要距離的獎勵值
# print("Reward values at specific distances:")
# print("Distance | Baseline | Coarse  | Fine    | Total")
# print("-" * 50)
# for dist in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
#     baseline = squashing_fn(dist, keypoint_coef_baseline[0], keypoint_coef_baseline[1])
#     coarse = squashing_fn(dist, keypoint_coef_coarse[0], keypoint_coef_coarse[1])
#     fine = squashing_fn(dist, keypoint_coef_fine[0], keypoint_coef_fine[1])
#     total = baseline + coarse + fine
#     print(f"{dist:.3f}    | {baseline:.4f}  | {coarse:.4f} | {fine:.4f} | {total:.4f}")