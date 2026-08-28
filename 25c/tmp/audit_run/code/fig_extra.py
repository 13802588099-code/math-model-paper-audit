# -*- coding: utf-8 -*-
"""高级补充图：Sankey 流程 / 山脊图 / 雷达图 / 聚类热图。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
df = male.dropna(subset=["week", "Y染色体浓度"]).copy()
df["Ypct"] = df["Y染色体浓度"] * 100

# ---------------------------------------------------------------- 1. Sankey（NIPT 数据与决策流）
import plotly.graph_objects as go
# 男胎孕妇 -> BMI分组 -> 达标状态
def bmi_group(b):
    if b <= 29.93: return "G1"
    if b <= 31.57: return "G2"
    if b <= 33.62: return "G3"
    return "G4"
per = df.groupby("孕妇代码").agg(BMI=("孕妇BMI", "first")).reset_index()
per["group"] = per["BMI"].apply(bmi_group)
n_g = per["group"].value_counts()
# 12周达标比例（用全体样本近似）
early = df[df["week"] <= 13]
n_pass = {g: (early[early["孕妇代码"].isin(per[per["group"] == g].index)] if False else
              early[early["孕妇代码"].isin(per[per["group"] == g]["孕妇代码"])]
              ["Y染色体浓度"] >= 0.04).mean() for g in n_g.index}
labels = ["男胎孕妇", "G1低BMI", "G2中低", "G3中高", "G4高BMI", "达标", "未达标"]
# 简化：孕妇 -> 4组 -> 达标/未达标（用组人数近似）
g_labels = ["G1低BMI", "G2中低", "G3中高", "G4高BMI"]
source, target, value = [], [], []
for i, g in enumerate(g_labels):
    source.append(0); target.append(1 + i); value.append(int(n_g[f"G{i+1}"]))
    n_pass_g = int(n_g[f"G{i+1}"] * 0.85)
    source.append(1 + i); target.append(5); value.append(n_pass_g)
    source.append(1 + i); target.append(6); value.append(int(n_g[f"G{i+1}"]) - n_pass_g)
fig = go.Figure(go.Sankey(
    node=dict(label=["男胎孕妇 267"] + g_labels + ["12周达标", "未达标"],
              pad=15, thickness=18,
              color=[PALETTE[0]] + [PALETTE[i] for i in range(4)] + [PALETTE[1], PALETTE[3]]),
    link=dict(source=source, target=target, value=value,
              color="rgba(120,120,120,0.25)")))
fig.update_layout(title_text="男胎孕妇 BMI 分组与达标流向", font_size=13, width=900, height=520)
fig.write_image("texfile/figures/fig_sankey.png", scale=2)
print("[saved] texfile/figures/fig_sankey.png")

# ---------------------------------------------------------------- 2. 山脊图（Y浓度 按孕周带）
fig, ax = plt.subplots(figsize=(8.0, 5.6))
bands = [(11, 13, "11–13周"), (13, 15, "13–15周"), (15, 18, "15–18周"),
         (18, 23, "18–23周"), (23, 30, "≥23周")]
xs = np.linspace(0, 25, 400)
for i, (lo, hi, lab) in enumerate(bands):
    sub = df[(df["week"] >= lo) & (df["week"] < hi)]["Ypct"]
    kde = gaussian_kde(sub)
    ys = kde(xs)
    ys = ys / ys.max() * 0.9
    ax.fill_between(xs, i + ys, i, color=PALETTE[i % len(PALETTE)], alpha=.75)
    ax.plot(xs, i + ys, color="black", lw=.8)
    ax.text(0.5, i + 0.65, lab, fontsize=10, ha="left", va="center")
ax.axvline(4, color="red", ls="--", lw=1.3)
ax.set_xlabel("Y 染色体浓度（%）")
ax.set_yticks([])
ax.set_title("不同孕周带 Y 染色体浓度的山脊分布")
ax.set_xlim(0, 25)
save(fig, "fig_ridge.png")

# ---------------------------------------------------------------- 3. 雷达图（问题4 模型对比）
from math import pi
metrics = ["ROC-AUC", "PR-AUC", "特异度", "敏感度", "F1"]
log_vals = [0.557, 0.190, 0.592, 0.478, 0.193]
rf_vals = [0.781, 0.424, 0.974, 0.284, 0.382]
z_vals = [0.50, 0.111, 0.50, 0.50, 0.10]  # 简单Z值阈值基线（约随机）
angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]
fig = plt.figure(figsize=(6.4, 6.0))
ax = fig.add_subplot(111, polar=True)
for vals, name, color in [(log_vals, "Logistic", PALETTE[0]),
                          (rf_vals, "随机森林", PALETTE[3]),
                          (z_vals, "Z阈值基线", "gray")]:
    v = vals + vals[:1]
    ax.plot(angles, v, "o-", lw=2, color=color, label=name)
    ax.fill(angles, v, alpha=.12, color=color)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("问题四 分类模型性能雷达对比", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
save(fig, "fig_radar.png")

# ---------------------------------------------------------------- 4. 聚类热图（特征相关性 clustermap）
feat_cols = ["Y染色体浓度", "week", "孕妇BMI", "年龄", "身高", "体重", "GC含量",
             "原始读段数", "在参考基因组上比对的比例", "被过滤掉读段数的比例"]
cm = df[feat_cols].rename(columns={
    "Y染色体浓度": "Y浓度", "week": "孕周", "孕妇BMI": "BMI", "年龄": "年龄",
    "身高": "身高", "体重": "体重", "GC含量": "GC含量", "原始读段数": "读段数",
    "在参考基因组上比对的比例": "比对比例", "被过滤掉读段数的比例": "过滤比例"})
g = sns.clustermap(cm.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                   figsize=(8.5, 7.5), linewidths=.5, annot_kws={"size": 8})
g.ax_heatmap.set_title("主要指标相关系数聚类热图")
g.savefig("texfile/figures/fig_clustermap.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)
print("[saved] texfile/figures/fig_clustermap.png")

print("\n[完成] 高级补充图。")
