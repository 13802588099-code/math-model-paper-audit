# -*- coding: utf-8 -*-
"""问题3：多因素达标时间 + 检测误差 + 达标比例 + 蒙特卡洛风险最小化。

1. 多因素混合效应：Y浓度 ~ week + BMI + 年龄（身高/体重与BMI共线，已并入BMI）。
2. 共线性诊断(VIF) + 标准化系数特征重要性。
3. 蒙特卡洛：以个体内线性插值的达标时间为基础，叠加检测误差 ε~N(0,σ_meas) 的
   扰动（误差通过斜率 b≈0.31%/周 传播为达标时间的不确定度），bootstrap 生成
   各 BMI 组的达标时间分布与达标比例曲线。
4. 最佳时点 = 组内 95% 孕妇达标的最早孕周（测序失败率<5%），使准确性与及时性均衡；
   并做检测误差 σ_meas 的敏感性分析。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
df = male.dropna(subset=["week", "Y染色体浓度"]).copy()
df["Ypct"] = df["Y染色体浓度"] * 100

# ---------------------------------------------------------------- 1. 多因素混合效应
me = sm.MixedLM.from_formula("Ypct ~ week + 孕妇BMI + 年龄", data=df,
                             groups=df["孕妇代码"])
r = me.fit(reml=True, method="lbfgs")
b0 = r.params["Intercept"]; b_week = r.params["week"]
b_bmi = r.params["孕妇BMI"]; b_age = r.params["年龄"]
sig_u = np.sqrt(r.cov_re.iloc[0, 0]); sig_e = np.sqrt(r.scale)
print("=== 多因素混合效应 Ypct ~ week + BMI + age ===")
print(f"  week={b_week:.4f}%/周(p={r.pvalues['week']:.2e})  "
      f"BMI={b_bmi:.4f}(p={r.pvalues['孕妇BMI']:.3g})  "
      f"age={b_age:.4f}(p={r.pvalues['年龄']:.3g})")
print(f"  σ_u(个体间)={sig_u:.3f}%  σ_e(个体内)={sig_e:.3f}%")

# 共线性 VIF
Xv = df[["孕妇BMI", "体重", "身高", "年龄"]].dropna()
Xv = sm.add_constant(Xv)
vif = pd.DataFrame({"变量": Xv.columns,
                    "VIF": [variance_inflation_factor(Xv.values, i)
                            for i in range(Xv.shape[1])]})
print("\n=== 协变量共线性 VIF（>10 表示严重共线）===")
print(vif.round(1).to_string(index=False))

# 标准化系数（特征重要性）
Xz = df[["孕妇BMI", "年龄", "体重", "身高", "week"]].copy()
for c in Xz.columns:
    Xz[c] = (Xz[c] - Xz[c].mean()) / Xz[c].std()
Xz = sm.add_constant(Xz)
yz = (df["Ypct"] - df["Ypct"].mean()) / df["Ypct"].std()
modz = sm.OLS(yz, Xz).fit()
print("\n=== 标准化系数（Y浓度，正=提升达标）===")
for c in ["孕妇BMI", "体重", "年龄", "身高", "week"]:
    print(f"  {c:6s} β*={modz.params[c]:+.3f}")

# ---------------------------------------------------------------- 2. 达标时间（插值）
def pass_week(g):
    g = g.sort_values("week")
    y = g["Y染色体浓度"].values; w = g["week"].values
    if y[0] >= 0.04:
        return w[0]
    idx = np.where(y >= 0.04)[0]
    if len(idx) == 0:
        return w[-1] + 4.0
    k = idx[0]
    return float(w[k - 1] + (0.04 - y[k - 1]) / (y[k] - y[k - 1]) * (w[k] - w[k - 1]))

rows = []
for code, g in df.groupby("孕妇代码"):
    rows.append({"孕妇代码": code, "达标时间": pass_week(g),
                 "BMI": g["孕妇BMI"].iloc[0], "年龄": g["年龄"].iloc[0],
                 "体重": g["体重"].iloc[0], "身高": g["身高"].iloc[0]})
per = pd.DataFrame(rows)

def bmi_group(b):
    if b <= 29.93: return "G1"
    if b <= 31.57: return "G2"
    if b <= 33.62: return "G3"
    return "G4"
per["group"] = per["BMI"].apply(bmi_group)
order = ["G1", "G2", "G3", "G4"]
colors = {g: PALETTE[i] for i, g in enumerate(order)}
grp_meta = {g: (per[per["group"] == g]["BMI"].min(),
                per[per["group"] == g]["BMI"].max()) for g in order}

# ---------------------------------------------------------------- 3. 蒙特卡洛
rng = np.random.default_rng(2025)
SLOPE = b_week          # 0.3125 %/周

def mc_dist(sig_meas, K=300):
    """每位孕妇达标时间 + 检测误差扰动，bootstrap 生成组内分布。"""
    out = {}
    for g in order:
        sub = per[per["group"] == g]
        T0 = sub["达标时间"].values
        # 检测误差 ε 传播为达标时间不确定度：δt = -ε/slope
        eps = rng.normal(0, sig_meas, (len(sub), K))
        Tsim = T0[:, None] - eps / SLOPE
        out[g] = Tsim.ravel()
    return out

def opt_timing(mc, alpha=0.95):
    weeks = np.arange(10, 30, 0.25)
    res = {}
    for g in order:
        T = mc[g]
        ratio = np.array([np.mean(T <= t) for t in weeks])
        hit = np.where(ratio >= alpha)[0]
        t_star = weeks[hit[0]] if len(hit) else weeks[-1]
        res[g] = (weeks, ratio, t_star)
    return res

# 主结果 σ_meas = 1.7%（个体内标准差上界）
mc = mc_dist(sig_e)
opt = opt_timing(mc)
print("\n=== 蒙特卡洛最佳时点（σ_meas=1.7%）===")
summary_rows = []
for g in order:
    weeks, ratio, t_star = opt[g]
    T = mc[g]
    summary_rows.append({"group": g, "BMI区间": f"[{grp_meta[g][0]:.1f},{grp_meta[g][1]:.1f}]",
                         "达标时间中位(周)": np.median(T),
                         "90%达标周": np.percentile(T, 90),
                         "95%达标周": np.percentile(T, 95),
                         "最佳时点(95%达标)": t_star})
summary = pd.DataFrame(summary_rows)
print(summary.round(2).to_string(index=False))
summary.to_csv("code/q3_summary.csv", index=False)

# ---------------------------------------------------------------- 4. 检测误差敏感性
print("\n=== 检测误差 σ_meas 敏感性：各组 95% 达标周 ===")
sig_grid = [0.5, 1.0, 1.7, 2.5]
sens = {}
for sg in sig_grid:
    mc_s = mc_dist(sg)
    o = opt_timing(mc_s)
    sens[sg] = {g: o[g][2] for g in order}
sens_df = pd.DataFrame(sens).rename(columns=lambda c: f"σ={c}%")
sens_df.index = order
print(sens_df.round(2).to_string())
print("\nσ_meas 0.5%->2.5% 最佳时点推迟(周)：")
print((sens_df["σ=2.5%"] - sens_df["σ=0.5%"]).round(2).to_string())
sens_df.to_csv("code/q3_sensitivity.csv")

# ---------------------------------------------------------------- 图形
# 图1：达标比例曲线
fig, ax = plt.subplots(figsize=(7.4, 5.0))
for g in order:
    weeks, ratio, t_star = opt[g]
    ax.plot(weeks, ratio, lw=2, color=colors[g], label=g)
    ax.axvline(t_star, color=colors[g], ls=":", lw=1, alpha=.7)
    ax.plot(t_star, 0.95, "o", color=colors[g], ms=5)
ax.axhline(0.95, color="gray", ls=":", lw=1)
ax.set_xlabel("孕周（周）"); ax.set_ylabel("达标比例 P(Y≥4%)")
ax.set_title("蒙特卡洛达标比例曲线（圆点=95%达标周）")
ax.legend(title="BMI分组", fontsize=9)
save(fig, "fig_q3_passratio.png")

# 图2：达标时间分布（小提琴 + 分位点）
fig, ax = plt.subplots(figsize=(7.2, 5.0))
data = [mc[g] for g in order]
parts = ax.violinplot(data, positions=np.arange(len(order)), showmeans=True,
                      showextrema=False, widths=0.8)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(colors[order[i]]); body.set_alpha(.6)
for i, g in enumerate(order):
    ax.scatter([i] * 60, np.percentile(mc[g], np.linspace(5, 95, 60)),
               s=6, color="black", alpha=.3)
ax.axhline(12, color="orange", ls=":", lw=1); ax.axhline(27, color="red", ls=":", lw=1)
ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order)
ax.set_xlabel("BMI 分组"); ax.set_ylabel("达标时间（周）")
ax.set_title("各组达标时间蒙特卡洛分布")
save(fig, "fig_q3_violin.png")

# 图3：最佳时点 + 敏感性误差棒
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xpos = np.arange(len(order))
yvals = [summary.loc[i, "最佳时点(95%达标)"] for i in range(len(order))]
err = (sens_df["σ=2.5%"] - sens_df["σ=0.5%"]).values / 2
ax.bar(xpos, yvals, color=[colors[g] for g in order], alpha=.85,
       yerr=err, capsize=5, label="最佳时点(95%达标)")
for i, g in enumerate(order):
    ax.text(i, yvals[i] + 0.3, f"BMI {grp_meta[g][0]:.0f}–{grp_meta[g][1]:.0f}",
            ha="center", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(order)
ax.set_xlabel("BMI 分组"); ax.set_ylabel("孕周（周）")
ax.set_title("各组最佳 NIPT 时点（误差棒=检测误差敏感性）")
ax.legend(fontsize=9)
save(fig, "fig_q3_optimal.png")

# 图4：特征重要性（标准化系数）
fig, ax = plt.subplots(figsize=(6.2, 4.0))
feats = ["体重", "孕妇BMI", "身高", "年龄", "week"]
vals = [modz.params[f] for f in feats]
names = ["体重", "BMI", "身高", "年龄", "孕周"]
bars = ax.barh(names, vals, color=[PALETTE[0] if v < 0 else PALETTE[2] for v in vals])
ax.axvline(0, color="gray", lw=1)
ax.set_xlabel("标准化回归系数 β*")
ax.set_title("Y浓度影响因素的标准化系数")
save(fig, "fig_q3_importance.png")

print("\n[完成] 问题3。")
