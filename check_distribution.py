#####################
# check_distribution.py
# Check the distribution of the data
# Author: JianXin Ji
# Date: 2024-07-26
# Version: 1.0
# Location: Harbin
#####################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 读取数据
hmu_df = pd.read_csv('HMU_ALL_PT_LABEL.csv')
tcga_df = pd.read_csv('TCGA_ALL_PT_LABEL.csv')
# import ipdb;ipdb.set_trace()
# 计算标签值分布
hmu_label_counts = hmu_df['label'].value_counts().sort_index()
tcga_label_counts = tcga_df['label'].value_counts().sort_index()

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))

# 创建柱状图
bar_width = 0.35
labels = hmu_label_counts.index

bar1 = ax.bar(labels - bar_width/2, hmu_label_counts, bar_width, label='HMU', alpha=0.7)
bar2 = ax.bar(labels + bar_width/2, tcga_label_counts, bar_width, label='TCGA', alpha=0.7)

# 绘制平滑曲线
def plot_smooth_curve(x, y, ax, label):
    x_new = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)
    ax.plot(x_new, y_smooth, label=label, linewidth=2)

# 获取柱状图的中心位置和高度
hmu_centers = [bar.get_x() + bar.get_width()/2 for bar in bar1]
hmu_heights = [bar.get_height() for bar in bar1]
tcga_centers = [bar.get_x() + bar.get_width()/2 for bar in bar2]
tcga_heights = [bar.get_height() for bar in bar2]

# 绘制平滑曲线
plot_smooth_curve(np.array(hmu_centers), np.array(hmu_heights), ax, label='HMU Trend')
plot_smooth_curve(np.array(tcga_centers), np.array(tcga_heights), ax, label='TCGA Trend')

# 添加标签和标题
ax.set_xlabel('Labels')
ax.set_ylabel('Counts')
ax.set_title('Label Distribution with Trend Lines')
ax.legend()

# 保存图形
plt.tight_layout()
plt.savefig('label_distribution_with_trend_lines.png')

# 关闭图形
plt.close()
