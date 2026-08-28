# -*- coding: utf-8 -*-
"""问题1：Y染色体浓度与孕周、BMI 等相关特性分析 + 关系模型 + 显著性检验。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
df = male.copy()

# Y浓度以百分数表示便于解释
df["Ypct"] = df["Y染色体浓度"] * 100

print("=" * 70)
print("问题1：描述性统计")
print("=" * 70)
desc = df["Ypct"].describe()
print(desc.round(4).to_string())
print(f"\nY浓度>=4% 样本占比: {(df['Ypct'] >= 4).mean():.4f}")

# ---------------------------------------------------------------- 相关性分析
print("\n" + "=" * 70)
print("问题1：Pearson / Spearman 相关性（与 Y染色体浓度）")
print("=" * 70)
features = ["week", "孕妇BMI", "年龄", "身高", "体重",
            "GC含量", "唯一比对的读段数", "原始读段数",
            "在参考基因组上比对的比例", "被过滤掉读段数的比例"]
corr_rows = []
for f in features:
    sub = df[[f, "Y染色体浓度"]].dropna()
    rp, pp = stats.pearsonr(sub[f], sub["Y染色体浓度"])
    rs, ps = stats.spearmanr(sub[f], sub["Y染色体浓度"])
    corr_rows.append({"指标": f, "Pearson_r": rp, "Pearson_p": pp,
                      "Spearman_rho": rs, "Spearman_p": ps})
    print(f"{f:26s} Pearson r={rp:+.4f} (p={pp:.2e})   "
          f"Spearman rho={rs:+.4f} (p={ps:.2e})")
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv("code/q1_corr.csv", index=False)

# 相关系数热力图（核心变量）
heat_vars = ["Y染色体浓度", "week", "孕妇BMI", "年龄", "身高", "体重", "GC含量"]
heat = df[heat_vars].rename(columns={
    "Y染色体浓度": "Y浓度", "week": "孕周", "孕妇BMI": "BMI",
    "年龄": "年龄", "身高": "身高", "体重": "体重", "GC含量": "GC含量"})
corr = heat.corr(method="pearson")
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(7.2, 6.0))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=.6, cbar_kws={"shrink": .8},
            annot_kws={"size": 10}, ax=ax)
ax.set_title("主要指标 Pearson 相关系数热力图", pad=12)
save(fig, "fig_q1_corr_heatmap.png")

# pairplot：Y浓度 vs 孕周/BMI/年龄
pair = df[["Y染色体浓度", "week", "孕妇BMI", "年龄"]].rename(columns={
    "Y染色体浓度": "Y浓度", "week": "孕周", "孕妇BMI": "BMI", "年龄": "年龄"})
g = sns.pairplot(pair, diag_kind="kde", corner=True,
                 plot_kws={"s": 12, "alpha": 0.45, "color": PALETTE[0]})
g.fig.suptitle("Y浓度与孕周、BMI、年龄的成对关系", y=1.02)
g.savefig("texfile/figures/fig_q1_pairplot.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)
print("[saved] texfile/figures/fig_q1_pairplot.png")

# ---------------------------------------------------------------- 关系模型
print("\n" + "=" * 70)
print("问题1：关系模型（statsmodels OLS）")
print("=" * 70)

# 模型 A：线性
mA = smf.ols("Ypct ~ week + 孕妇BMI", data=df).fit()
# 模型 B：多项式 + 交互
df["week2"] = df["week"] ** 2
df["BMI2"] = df["孕妇BMI"] ** 2
df["week_BMI"] = df["week"] * df["孕妇BMI"]
mB = smf.ols("Ypct ~ week + week2 + 孕妇BMI + BMI2 + week_BMI", data=df).fit()
# 模型 C：对数线性
df["lnY"] = np.log(df["Y染色体浓度"])
mC = smf.ols("lnY ~ week + 孕妇BMI", data=df).fit()

models = [("模型A 线性", mA, "Ypct"), ("模型B 多项式", mB, "Ypct"),
          ("模型C 对数线性", mC, "lnY")]
for name, m, _ in models:
    print(f"\n--- {name} ---")
    print(f"R^2 = {m.rsquared:.4f}  adj.R^2 = {m.rsquared_adj:.4f}  "
          f"AIC = {m.aic:.1f}  BIC = {m.bic:.1f}  F = {m.fvalue:.1f} (p={m.f_pvalue:.2e})")
    print(m.params.round(4).to_string())

print("\n=== 模型B 系数显著性汇总 ===")
print(mB.summary().tables[1])

# 保存模型B完整结果
with open("code/q1_modelB_summary.txt", "w") as f:
    f.write(mB.summary().as_text())

# 最优模型（多项式）预测，绘制拟合与残差
df["yhat_B"] = mB.predict(df)
df["resid_B"] = df["Ypct"] - df["yhat_B"]

# ---------------------------------------------------------------- 图形
# 图：Y浓度随孕周变化（BMI分组着色）+ 拟合曲线
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

# (a) Y vs week
bmi_bins = pd.cut(df["孕妇BMI"], bins=[20, 28, 32, 36, 48],
                  labels=["BMI 20–28", "BMI 28–32", "BMI 32–36", "BMI ≥36"])
for i, (lab, sub) in enumerate(df.groupby(bmi_bins, observed=True)):
    axes[0].scatter(sub["week"], sub["Ypct"], s=16, alpha=0.5,
                    color=PALETTE[i], label=lab)
# 全体平均趋势（分箱均值）
wbin = pd.cut(df["week"], bins=np.arange(11, 30, 2))
means = df.groupby(wbin, observed=True)["Ypct"].agg(["mean", "sem"])
wc = [iv.mid for iv in means.index]
axes[0].errorbar(wc, means["mean"], yerr=1.96 * means["sem"], fmt="o-",
                 color="black", lw=1.6, ms=4, capsize=3,
                 label="均值±95%置信带")
axes[0].axhline(4, color="red", ls="--", lw=1.2, label="4% 达标阈值")
axes[0].set_xlabel("孕周（周）")
axes[0].set_ylabel("Y 染色体浓度（%）")
axes[0].set_title("(a) Y浓度随孕周的变化")
axes[0].legend(fontsize=8, loc="upper left")

# (b) Y vs BMI
for i, (lab, sub) in enumerate(df.groupby(pd.cut(df["week"], bins=[11, 14, 18, 23, 30],
                                                 labels=["11–14w", "14–18w", "18–23w", "≥23w"]),
                                             observed=True)):
    axes[1].scatter(sub["孕妇BMI"], sub["Ypct"], s=16, alpha=0.5,
                    color=PALETTE[i], label=lab)
axes[1].axhline(4, color="red", ls="--", lw=1.2)
axes[1].set_xlabel("孕妇 BMI")
axes[1].set_ylabel("Y 染色体浓度（%）")
axes[1].set_title("(b) Y浓度随BMI的变化")
axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
save(fig, "fig_q1_scatter.png")

# 图：多项式模型拟合面（week×BMI 的等高线）
fig = plt.figure(figsize=(6.4, 5.2))
ax = fig.add_subplot(111)
wk = np.linspace(11, 29, 60)
bmi = np.linspace(21, 47, 60)
W, B = np.meshgrid(wk, bmi)
pred = mB.params["Intercept"] + mB.params["week"] * W + mB.params["week2"] * W**2 \
    + mB.params["孕妇BMI"] * B + mB.params["BMI2"] * B**2 \
    + mB.params["week_BMI"] * W * B
cs = ax.contourf(W, B, pred, levels=20, cmap="viridis")
ax.contour(W, B, pred, levels=[4], colors="red", linewidths=1.8,
           linestyles="--")
ax.clabel(ax.contour(W, B, pred, levels=[4, 6, 8, 10], colors="white",
                     linewidths=0.6), fmt="%.0f%%", fontsize=8)
ax.scatter(df["week"], df["孕妇BMI"], s=8, alpha=0.25, color="black")
fig.colorbar(cs, ax=ax, label="Y浓度预测值（%）")
ax.set_xlabel("孕周（周）")
ax.set_ylabel("BMI")
ax.set_title("多项式模型拟合：Y浓度等值线（红线=4%达标）")
save(fig, "fig_q1_contour.png")

# 残差诊断
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
sns.scatterplot(x=df["yhat_B"], y=df["resid_B"], s=12, alpha=.4,
                color=PALETTE[0], ax=axes[0])
axes[0].axhline(0, color="red", ls="--", lw=1)
axes[0].set_xlabel("拟合值")
axes[0].set_ylabel("残差")
axes[0].set_title("(a) 残差 vs 拟合值")

sm.graphics.qqplot(df["resid_B"], line="45", ax=axes[1], alpha=.3)
axes[1].set_title("(b) 残差正态 Q-Q 图")

sns.histplot(df["resid_B"], kde=True, color=PALETTE[1], ax=axes[2])
axes[2].set_xlabel("残差")
axes[2].set_title("(c) 残差分布")
fig.tight_layout()
save(fig, "fig_q1_resid.png")

print("\n[完成] 问题1 结果已输出。")
print("模型B 关键参数：")
for k in ["week", "week2", "孕妇BMI", "BMI2", "week_BMI"]:
    print(f"  {k:10s} coef={mB.params[k]:+.4f}  p={mB.pvalues[k]:.2e}")
