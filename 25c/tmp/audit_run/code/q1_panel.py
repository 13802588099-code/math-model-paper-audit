# -*- coding: utf-8 -*-
"""问题1 补充：个体内（面板）分析与混合效应模型，刻画 Y浓度 随孕周的个体内上升趋势。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
df = male.copy()
df = df.dropna(subset=["week", "Y染色体浓度"]).reset_index(drop=True)
df["Ypct"] = df["Y染色体浓度"] * 100

# ---------------------------------------------------------------- 个体内斜率
print("=" * 70)
print("个体内斜率：每位孕妇 Y浓度 对 孕周 的线性斜率")
print("=" * 70)
slopes = []
for code, g in df.groupby("孕妇代码"):
    g = g.sort_values("week")
    if len(g) >= 2 and g["week"].nunique() >= 2:
        b = np.polyfit(g["week"], g["Ypct"], 1)[0]
        slopes.append({"孕妇代码": code, "slope": b, "BMI": g["孕妇BMI"].iloc[0],
                       "n": len(g)})
sl = pd.DataFrame(slopes)
print(f"可估计斜率的孕妇数: {len(sl)}（占总 {df['孕妇代码'].nunique()} 的 "
      f"{len(sl)/df['孕妇代码'].nunique():.1%}）")
print("个体内斜率统计：")
print(sl["slope"].describe().round(4).to_string())
print(f"斜率>0 占比: {(sl['slope']>0).mean():.3f}   "
      f"斜率均值 = {sl['slope'].mean():.3f} %/周 (p="
      f"{pd.Series(sl['slope']).pipe(lambda s: __import__('scipy').stats.ttest_1samp(s, 0).pvalue):.2e})")

# 个体内斜率 vs BMI 的关系
r_sl_bmi = np.corrcoef(sl["slope"], sl["BMI"])[0, 1]
print(f"个体内斜率 与 BMI 相关系数 r = {r_sl_bmi:.3f}")

# ---------------------------------------------------------------- 混合效应模型
print("\n" + "=" * 70)
print("混合效应模型：Ypct ~ week + BMI + (1|孕妇代码)")
print("=" * 70)
import statsmodels.api as sm
# 编码孕妇代码为分类
df["code_cat"] = df["孕妇代码"].astype("category")
me = sm.MixedLM.from_formula("Ypct ~ week + 孕妇BMI", data=df,
                             groups=df["孕妇代码"])
mer = me.fit(reml=True, method="lbfgs")
print(mer.summary())
with open("code/q1_mixed_summary.txt", "w") as f:
    f.write(mer.summary().as_text())

# 个体间方差 vs 个体内方差
print(f"\n随机截距方差: {mer.cov_re.iloc[0,0]:.4f}  残差方差: {mer.scale:.4f}")
icc = mer.cov_re.iloc[0, 0] / (mer.cov_re.iloc[0, 0] + mer.scale)
print(f"组内相关系数 ICC = {icc:.3f}（说明个体间差异占比 {icc:.1%}）")

# ---------------------------------------------------------------- 图形
# 图：面条图（随机抽取 40 名孕妇的 Y浓度-孕周 轨迹）
fig, ax = plt.subplots(figsize=(7.0, 5.0))
rng = np.random.default_rng(42)
sample_codes = rng.choice(df["孕妇代码"].unique(), size=40, replace=False)
for i, code in enumerate(sample_codes):
    g = df[df["孕妇代码"] == code].sort_values("week")
    ax.plot(g["week"], g["Ypct"], "-o", ms=3, lw=1, alpha=.5,
            color=PALETTE[i % len(PALETTE)])
ax.axhline(4, color="red", ls="--", lw=1.3, label="4% 达标阈值")
ax.set_xlabel("孕周（周）")
ax.set_ylabel("Y 染色体浓度（%）")
ax.set_title("随机 40 名男胎孕妇的 Y浓度-孕周 个体轨迹")
ax.legend()
save(fig, "fig_q1_spaghetti.png")

# 图：个体内斜率分布 + 斜率 vs BMI
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
sns.histplot(sl["slope"], kde=True, color=PALETTE[0], ax=axes[0])
axes[0].axvline(sl["slope"].mean(), color="red", ls="--",
                label=f"均值 {sl['slope'].mean():.3f} %/周")
axes[0].axvline(0, color="gray", ls=":", lw=1)
axes[0].set_xlabel("个体内斜率（%/周）")
axes[0].set_ylabel("频数")
axes[0].set_title("(a) 个体内斜率分布")
axes[0].legend()
sns.scatterplot(data=sl, x="BMI", y="slope", s=20, alpha=.5,
                color=PALETTE[1], ax=axes[1])
sns.regplot(data=sl, x="BMI", y="slope", scatter=False, ax=axes[1],
            color="black", line_kws={"ls": "--"})
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("个体内斜率（%/周）")
axes[1].set_title("(b) 个体内斜率 与 BMI 的关系")
fig.tight_layout()
save(fig, "fig_q1_within_slope.png")

print("\n[完成] 面板/混合效应分析。")
