import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
methods = ['Victim Model', 'Method A', 'Method B', 'Our Method']
clean_accuracy = [89.1, 87.2, 85.4, 88.5]  # 干净准确率 (%)
attack_success_rate = [99.2, 45.3, 28.7, 3.1]  # 攻击成功率 (%)

# 转换为百分比小数
clean_accuracy = [x/100 for x in clean_accuracy]
attack_success_rate = [x/100 for x in attack_success_rate]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 设置条形图位置和宽度
x = np.arange(len(methods))
width = 0.35

# 绘制条形图
bars1 = ax.bar(x - width/2, clean_accuracy, width, label='Clean Accuracy',
               color='#2E86AB', edgecolor='black', linewidth=0.5, alpha=0.8)
bars2 = ax.bar(x + width/2, attack_success_rate, width, label='Attack Success Rate (ASR)',
               color='#A23B72', edgecolor='black', linewidth=0.5, alpha=0.8)

# 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# 设置图表标题和标签
ax.set_xlabel('Training Methods', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Metrics', fontsize=14, fontweight='bold')
ax.set_title('Clean Accuracy vs Attack Success Rate by Training Method\n(Defense Effectiveness Comparison)',
             fontsize=16, fontweight='bold', pad=20)

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')

# 设置y轴
ax.set_ylim(0, 1.1)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.2)], fontsize=11)

# 添加图例
ax.legend(fontsize=12, loc='upper left')

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加说明文本
ax.text(0.02, 0.98, 'Our method maintains high clean accuracy\nwhile significantly reducing ASR',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.2))

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('defense_comparison_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('defense_comparison_chart.pdf', bbox_inches='tight')

# 显示图像
plt.show()

# 打印统计结果
print("=" * 60)
print("DEFENSE EFFECTIVENESS COMPARISON")
print("=" * 60)
for i, method in enumerate(methods):
    print(f"{method:15} | Clean Accuracy: {clean_accuracy[i]:.3f} | ASR: {attack_success_rate[i]:.3f}")

print("\n" + "=" * 60)
print("KEY FINDINGS:")
print("=" * 60)
print("• Our method reduces ASR from 99.2% to 3.1% (96.1% reduction)")
print("• Maintains clean accuracy at 88.5% (vs 89.1% original)")
print("• Outperforms Method A (45.3% ASR) and Method B (28.7% ASR)")
print("• Demonstrates effective defense with minimal impact on utility")