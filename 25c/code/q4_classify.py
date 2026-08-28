# -*- coding: utf-8 -*-
"""问题4：女胎异常判定分类（Logistic 回归 与 随机森林 对比）。

标签：染色体的非整倍体(AB列)非空 = 异常(1)，空 = 正常(0)。
特征：X染色体Z值、13/18/21号Z值、X染色体浓度、GC含量(总/各染色体)、
读段数及比例、孕妇BMI、年龄、孕周，并构造 maxZ、minZ 等衍生特征。
处理类别不平衡(class_weight)，5 折交叉验证评估 ROC/PR/混淆矩阵/特征重要性。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             classification_report)
from common import setup_style, save, load_data, PALETTE

setup_style()
male, female = load_data()
f = female.copy()

# ---------------------------------------------------------------- 特征工程
feat = ["X染色体的Z值", "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值",
        "X染色体浓度", "GC含量", "13号染色体的GC含量", "18号染色体的GC含量",
        "21号染色体的GC含量", "原始读段数", "唯一比对的读段数",
        "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例",
        "孕妇BMI", "年龄", "week"]
f = f.dropna(subset=feat).reset_index(drop=True)
# 衍生特征
f["maxZ"] = f[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].max(axis=1)
f["minZ"] = f[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].min(axis=1)
f["log读段数"] = np.log10(f["唯一比对的读段数"] + 1)
f["Z极差"] = f["maxZ"] - f["minZ"]
feat_full = feat + ["maxZ", "minZ", "log读段数", "Z极差"]

X = f[feat_full].values
y = f["异常"].values
print(f"样本数={len(f)}  异常={y.sum()} ({(y.sum()/len(y)):.1%})  正常={(y==0).sum()}")
print(f"特征数={len(feat_full)}")

# ---------------------------------------------------------------- 逻辑斯蒂
print("\n=== Logistic 回归（5折交叉验证）===")
log = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scoring = {"auc": "roc_auc", "ap": "average_precision", "f1": "f1"}
res_log = cross_validate(log, X, y, cv=cv, scoring=scoring)
y_prob_log = cross_val_predict(log, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred_log = cross_val_predict(log, X, y, cv=cv)
for k in scoring:
    print(f"  {k}: {res_log[f'test_{k}'].mean():.4f} ± {res_log[f'test_{k}'].std():.4f}")

# ---------------------------------------------------------------- 随机森林
print("\n=== 随机森林（5折交叉验证）===")
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                            max_depth=None, random_state=42, n_jobs=-1)
res_rf = cross_validate(rf, X, y, cv=cv, scoring=scoring)
y_prob_rf = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
for k in scoring:
    print(f"  {k}: {res_rf[f'test_{k}'].mean():.4f} ± {res_rf[f'test_{k}'].std():.4f}")

print("\n=== Logistic 混淆矩阵 ===")
print(confusion_matrix(y, y_pred_log))
print(classification_report(y, y_pred_log, target_names=["正常", "异常"], digits=3))
print("=== 随机森林 混淆矩阵 ===")
print(confusion_matrix(y, y_pred_rf))
print(classification_report(y, y_pred_rf, target_names=["正常", "异常"], digits=3))

# ---------------------------------------------------------------- 特征重要性
rf.fit(X, y)
imp = pd.DataFrame({"特征": feat_full, "重要性": rf.feature_importances_}).sort_values(
    "重要性", ascending=False)
print("\n=== 随机森林特征重要性（Top 12）===")
print(imp.head(12).round(4).to_string(index=False))

log.fit(X, y)
coef = pd.DataFrame({"特征": feat_full, "系数": log.coef_[0]}).sort_values(
    "系数", key=abs, ascending=False)
print("\n=== Logistic 系数（Top 12，按|系数|）===")
print(coef.head(12).round(4).to_string(index=False))

# ---------------------------------------------------------------- 图形
# 图1：ROC + PR 曲线对比
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
for name, prob, color in [("Logistic", y_prob_log, PALETTE[0]),
                          ("随机森林", y_prob_rf, PALETTE[3])]:
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
    pre, rec, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    axes[1].plot(rec, pre, lw=2, color=color, label=f"{name} (AP={ap:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("假阳性率 FPR"); axes[0].set_ylabel("真阳性率 TPR")
axes[0].set_title("(a) ROC 曲线"); axes[0].legend()
axes[1].axhline(y.mean(), color="gray", ls=":", lw=1)
axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
axes[1].set_title("(b) PR 曲线"); axes[1].legend()
fig.tight_layout()
save(fig, "fig_q4_roc_pr.png")

# 图2：混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
for ax, cm, name in [(axes[0], confusion_matrix(y, y_pred_log), "Logistic"),
                     (axes[1], confusion_matrix(y, y_pred_rf), "随机森林")]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["正常", "异常"], yticklabels=["正常", "异常"],
                annot_kws={"size": 13}, cbar=False)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title(f"{name}")
fig.tight_layout()
save(fig, "fig_q4_confusion.png")

# 图3：特征重要性
fig, ax = plt.subplots(figsize=(6.6, 4.6))
top = imp.head(10).iloc[::-1]
ax.barh(top["特征"], top["重要性"], color=PALETTE[0])
ax.set_xlabel("特征重要性")
ax.set_title("随机森林特征重要性（Top 10）")
fig.tight_layout()
save(fig, "fig_q4_importance.png")

# 图4：关键 Z 值在正常/异常间的分布（小提琴）
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
for ax, col, ttl in [(axes[0], "maxZ", "三染色体最大Z值"),
                     (axes[1], "X染色体浓度", "X染色体浓度")]:
    data = [f.loc[f["异常"] == 0, col], f.loc[f["异常"] == 1, col]]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE[i]); body.set_alpha=.6
    ax.set_xticks([0, 1]); ax.set_xticklabels(["正常", "异常"])
    ax.set_title(ttl)
fig.tight_layout()
save(fig, "fig_q4_zdist.png")

print("\n[完成] 问题4。")
