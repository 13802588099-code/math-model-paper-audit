# -*- coding: utf-8 -*-
"""问题2：男胎 BMI 决策树分段 + 各组最佳 NIPT 时点 + 检测误差(阈值)敏感性。

达标时间 = 首次 Y浓度>=4% 的孕周，个体内线性插值；
左删失(首次即达标)取首周为上界，右删失(窗内未达标)取末次孕周+4周为保守下界。
决策树在 BMI 上自动分段；最佳时点 = 组内 95% 孕妇达标的最早孕周；
敏感性分析对 4% 阈值施加 ±0.3% 扰动，重新计算各组最佳时点偏移。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
df = male.dropna(subset=["week", "Y染色体浓度"]).copy()
df["Ypct"] = df["Y染色体浓度"] * 100


def pass_week(g, thr=0.04):
    g = g.sort_values("week")
    y = g["Y染色体浓度"].values
    w = g["week"].values
    if y[0] >= thr:
        return w[0], "left"
    idx = np.where(y >= thr)[0]
    if len(idx) == 0:
        return w[-1] + 4.0, "right"
    k = idx[0]
    t = w[k - 1] + (thr - y[k - 1]) / (y[k] - y[k - 1]) * (w[k] - w[k - 1])
    return float(t), "hit"


def build(df, thr=0.04, max_depth=2, min_leaf=60, seed=42):
    rows = []
    for code, g in df.groupby("孕妇代码"):
        t, st = pass_week(g, thr)
        rows.append({"孕妇代码": code, "达标时间": t, "status": st,
                     "BMI": g["孕妇BMI"].iloc[0], "年龄": g["年龄"].iloc[0],
                     "身高": g["身高"].iloc[0], "体重": g["体重"].iloc[0]})
    per = pd.DataFrame(rows)
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf,
                                 random_state=seed)
    tree.fit(per[["BMI"]].values, per["达标时间"].values)
    per["leaf"] = tree.apply(per[["BMI"]].values)
    leaf_bmi = per.groupby("leaf")["BMI"].mean().sort_values()
    gmap = {leaf: f"G{i+1}" for i, leaf in enumerate(leaf_bmi.index)}
    per["group"] = per["leaf"].map(gmap)
    return per, tree


per, tree = build(df)
order = sorted(per["group"].unique())
colors = {g: PALETTE[i] for i, g in enumerate(order)}

summary = per.groupby("group").agg(
    n=("达标时间", "size"),
    BMI_min=("BMI", "min"), BMI_max=("BMI", "max"), BMI_med=("BMI", "median"),
    t_median=("达标时间", "median"),
    t_p90=("达标时间", lambda s: np.percentile(s, 90)),
    t_p95=("达标时间", lambda s: np.percentile(s, 95)),
).reset_index().sort_values("group")
print("达标时间全样本：")
print(per["达标时间"].describe().round(2).to_string())
print("status：", per["status"].value_counts().to_dict())
print("\n决策树分段（BMI）结果：")
print(summary.round(2).to_string(index=False))
summary.to_csv("code/q2_groups.csv", index=False)

# 每组在 12/14/16 周的达标比例（用该组全部样本）
print("\n各组在参考孕周的达标比例：")
for _, r in summary.iterrows():
    sub = df[df["孕妇代码"].isin(per[per["group"] == r["group"]]["孕妇代码"])]
    line = [f"{r['group']} BMI[{r['BMI_min']:.1f},{r['BMI_max']:.1f}]"]
    for wk in [12, 14, 16]:
        s = sub[np.abs(sub["week"] - wk) <= 1.0]
        line.append(f"{wk}周达标={ (s['Y染色体浓度']>=0.04).mean() if len(s)>5 else np.nan:.2f}")
    print("  ".join(line))

# ---------------------------------------------------------------- 阈值敏感性
# 用 4.0% 决策树确定的分组边界作为固定分组，再对阈值 ±0.3% 重新计算各组 P90/P95
print("\n=== 阈值 4%±0.3% 敏感性（固定分组边界）===")
bounds = sorted(np.round(tree.tree_.threshold[tree.tree_.threshold > 0], 2))
print("固定 BMI 分组边界：", bounds)
# 用决策树叶节点的 BMI 范围更稳：直接复用 per 的 group 边界
def assign_group(bmi, per):
    # 按每组 BMI 区间中点到边界切分
    ranges = {}
    for g, sub in per.groupby("group"):
        ranges[g] = (sub["BMI"].min(), sub["BMI"].max())
    for g, (lo, hi) in sorted(ranges.items()):
        if lo <= bmi <= hi:
            return g
    return None
per["_g"] = per["group"]

sens_rows = {}
for thr_pct in [3.7, 4.0, 4.3]:
    rows = []
    for code, g in df.groupby("孕妇代码"):
        t, st = pass_week(g, thr_pct / 100)
        rows.append({"孕妇代码": code, "达标时间": t, "BMI": g["孕妇BMI"].iloc[0]})
    per_t = pd.DataFrame(rows).merge(per[["孕妇代码", "group"]], on="孕妇代码")
    s_t = per_t.groupby("group")["达标时间"].agg(
        t_p90=lambda s: np.percentile(s, 90), t_p95=lambda s: np.percentile(s, 95))
    sens_rows[thr_pct] = s_t["t_p90"]
sens = pd.DataFrame(sens_rows).rename(columns={3.7: "阈值3.7%", 4.0: "阈值4.0%", 4.3: "阈值4.3%"})
sens = sens.reindex(order)
print("各组 90% 达标周：")
print(sens.round(2).to_string())
print("\n阈值 4.0%->3.7%(低估0.3%) 最佳时点提前(周)：")
print((sens["阈值3.7%"] - sens["阈值4.0%"]).round(2).to_string())
print("阈值 4.0%->4.3%(高估0.3%) 最佳时点推迟(周)：")
print((sens["阈值4.3%"] - sens["阈值4.0%"]).round(2).to_string())
sens.to_csv("code/q2_sensitivity.csv")

# ---------------------------------------------------------------- 图形
# 图1：达标比例随孕周（按BMI组，经验分箱）
fig, ax = plt.subplots(figsize=(7.2, 5.0))
bmi_bins = pd.cut(df["孕妇BMI"], bins=[20, 28, 32, 36, 48],
                  labels=["BMI 20–28", "28–32", "32–36", "≥36"])
for i, (lab, sub) in enumerate(df.groupby(bmi_bins, observed=True)):
    wbins = np.arange(11, 30, 1)
    p, wk = [], []
    for j in range(len(wbins) - 1):
        seg = sub[(sub["week"] >= wbins[j]) & (sub["week"] < wbins[j + 1])]
        if len(seg) >= 8:
            p.append((seg["Y染色体浓度"] >= 0.04).mean())
            wk.append((wbins[j] + wbins[j + 1]) / 2)
    ax.plot(wk, p, "-o", ms=4, lw=1.6, color=PALETTE[i], label=lab)
ax.axhline(0.95, color="gray", ls=":", lw=1)
ax.set_xlabel("孕周（周）"); ax.set_ylabel("达标比例 P(Y≥4%)")
ax.set_title("不同 BMI 组达标比例随孕周的变化")
ax.legend(fontsize=9)
save(fig, "fig_q2_passprob.png")

# 图2：达标时间随BMI分布 + 决策树分段 + 箱线图
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
sns.scatterplot(data=per, x="BMI", y="达标时间", hue="group", palette=colors,
                s=30, alpha=.7, ax=axes[0])
for thr in np.round(tree.tree_.threshold[tree.tree_.threshold > 0], 1):
    axes[0].axvline(thr, color="gray", ls=":", lw=1)
axes[0].axhline(12, color="orange", ls=":", lw=1, alpha=.7)
axes[0].axhline(27, color="orange", ls=":", lw=1, alpha=.7)
axes[0].set_xlabel("BMI"); axes[0].set_ylabel("达标时间（周）")
axes[0].set_title("(a) 达标时间随BMI分布与决策树分段")
axes[0].legend(fontsize=8)

sns.boxplot(data=per, x="group", y="达标时间", order=order,
            hue="group", palette=colors, legend=False, ax=axes[1])
sns.stripplot(data=per, x="group", y="达标时间", order=order, color="black",
              size=2.5, alpha=.4, ax=axes[1])
axes[1].set_xlabel("BMI 分组"); axes[1].set_ylabel("达标时间（周）")
axes[1].set_title("(b) 各组达标时间箱线图")
fig.tight_layout()
save(fig, "fig_q2_boxplot.png")

# 图3：决策树结构
fig, ax = plt.subplots(figsize=(8.5, 4.6))
plot_tree(tree, feature_names=["BMI"], filled=True, rounded=True,
          impurity=False, proportion=False, fontsize=9, precision=1, ax=ax)
ax.set_title("BMI 达标时间决策树")
save(fig, "fig_q2_tree.png")

# 图4：各组最佳时点（90%达标周）+ 敏感性误差棒
fig, ax = plt.subplots(figsize=(7.4, 4.6))
xpos = np.arange(len(order))
yvals = [summary.loc[summary["group"] == g, "t_p90"].iloc[0] for g in order]
err_low = [sens.loc[g, "阈值3.7%"] - sens.loc[g, "阈值4.0%"] for g in order]
err_high = [sens.loc[g, "阈值4.3%"] - sens.loc[g, "阈值4.0%"] for g in order]
ax.bar(xpos, yvals, color=[colors[g] for g in order], alpha=.85,
       yerr=[np.abs(err_low), np.abs(err_high)], capsize=4,
       label="最佳时点(90%达标)")
ax.plot(xpos, [summary.loc[summary["group"] == g, "t_median"].iloc[0] for g in order],
        "kD", ms=7, label="中位达标时间")
for i, g in enumerate(order):
    r = summary[summary["group"] == g].iloc[0]
    ax.text(i, yvals[i] + 0.3, f"BMI {r['BMI_min']:.0f}–{r['BMI_max']:.0f}",
            ha="center", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(order)
ax.set_xlabel("BMI 分组"); ax.set_ylabel("孕周（周）")
ax.set_title("各组最佳 NIPT 时点（误差棒=阈值±0.3%）")
ax.legend(fontsize=9)
save(fig, "fig_q2_optimal.png")

print("\n[完成] 问题2。")
