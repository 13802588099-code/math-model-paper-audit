from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "附件.xlsx"


def parse_week(value):
    match = re.match(r"(\d+)\s*w\s*\+?\s*(\d+)?", str(value).strip().replace("周", ""))
    if not match:
        return np.nan
    return int(match.group(1)) + int(match.group(2) or 0) / 7


male = pd.read_excel(DATA, sheet_name="男胎检测数据")
female = pd.read_excel(DATA, sheet_name="女胎检测数据")
for frame in (male, female):
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["week"] = frame["检测孕周"].map(parse_week)
    frame["异常"] = frame["染色体的非整倍体"].notna().astype(int)

male = male.dropna(subset=["week", "Y染色体浓度"]).copy()
male["Ypct"] = male["Y染色体浓度"] * 100

out = {}
out["data"] = {
    "male_rows": len(male),
    "male_women": int(male["孕妇代码"].nunique()),
    "female_rows": len(female),
    "female_women": int(female["孕妇代码"].nunique()),
    "male_rows_per_woman": male.groupby("孕妇代码").size().describe().round(3).to_dict(),
    "female_rows_per_woman": female.groupby("孕妇代码").size().describe().round(3).to_dict(),
}

# Technical replicates: same woman, draw number and gestational week.
tech_keys = ["孕妇代码", "检测抽血次数", "week"]
tech_sizes = male.groupby(tech_keys, dropna=False).size()
tech_groups = tech_sizes[tech_sizes > 1]
out["data"]["technical_replicate_groups"] = int(len(tech_groups))
out["data"]["technical_replicate_rows"] = int(tech_groups.sum())
out["data"]["unique_male_draws"] = int(len(tech_sizes))

ss = 0.0
dfree = 0
ranges = []
for _, group in male.groupby(tech_keys, dropna=False):
    if len(group) > 1:
        values = group["Ypct"].to_numpy()
        ss += float(((values - values.mean()) ** 2).sum())
        dfree += len(values) - 1
        ranges.append(float(values.max() - values.min()))
tech_sd = float(np.sqrt(ss / dfree)) if dfree else np.nan
out["measurement_error"] = {
    "pooled_technical_sd_pct_points": tech_sd,
    "technical_group_range_median_pct_points": float(np.median(ranges)),
    "technical_group_range_p90_pct_points": float(np.percentile(ranges, 90)),
    "technical_df": int(dfree),
}

# Assumption checks: static covariates and non-monotone threshold trajectories.
within_variation = {}
for col in ["孕妇BMI", "年龄", "身高", "体重"]:
    within_variation[col] = int((male.groupby("孕妇代码")[col].nunique(dropna=True) > 1).sum())
out["data"]["women_with_within_person_covariate_changes"] = within_variation

nonmonotone = 0
first_above_then_below = 0
for _, group in male.groupby("孕妇代码"):
    by_week = group.groupby("week")["Y染色体浓度"].mean().sort_index()
    vals = by_week.to_numpy()
    if len(vals) > 1 and np.any(np.diff(vals) < 0):
        nonmonotone += 1
    hits = np.flatnonzero(vals >= 0.04)
    if len(hits) and np.any(vals[hits[0] + 1 :] < 0.04):
        first_above_then_below += 1
out["data"]["women_with_any_decrease_in_mean_trajectory"] = nonmonotone
out["data"]["women_above_4pct_then_later_below"] = first_above_then_below

# Refit the mixed model on raw records and on technical-draw means.
def fit_mixed(frame):
    model = sm.MixedLM.from_formula(
        "Ypct ~ week + 孕妇BMI", data=frame, groups=frame["孕妇代码"]
    ).fit(reml=True, method="lbfgs")
    return {
        "n": int(model.nobs),
        "week_beta": float(model.params["week"]),
        "week_p": float(model.pvalues["week"]),
        "bmi_beta": float(model.params["孕妇BMI"]),
        "bmi_p": float(model.pvalues["孕妇BMI"]),
        "resid_sd": float(np.sqrt(model.scale)),
        "random_intercept_sd": float(np.sqrt(model.cov_re.iloc[0, 0])),
    }


raw_mixed = fit_mixed(male)
draw = (
    male.groupby(tech_keys, dropna=False)
    .agg(Ypct=("Ypct", "mean"), 孕妇BMI=("孕妇BMI", "first"))
    .reset_index()
)
draw_mixed = fit_mixed(draw)
out["mixed_models"] = {"raw_records": raw_mixed, "technical_draw_means": draw_mixed}


def pass_week(group, right_add=4.0):
    # Average technical replicates at the same week before determining crossings.
    grouped = group.groupby("week")["Y染色体浓度"].mean().sort_index()
    w = grouped.index.to_numpy(float)
    y = grouped.to_numpy(float)
    if y[0] >= 0.04:
        return float(w[0]), "left"
    idx = np.flatnonzero(y >= 0.04)
    if len(idx) == 0:
        return float(w[-1] + right_add), "right"
    k = idx[0]
    if y[k] == y[k - 1]:
        return float(w[k]), "hit"
    value = w[k - 1] + (0.04 - y[k - 1]) / (y[k] - y[k - 1]) * (w[k] - w[k - 1])
    return float(value), "hit"


def person_times(right_add=4.0):
    rows = []
    for code, group in male.groupby("孕妇代码"):
        timing, status = pass_week(group, right_add=right_add)
        rows.append(
            {
                "孕妇代码": code,
                "time": timing,
                "status": status,
                "BMI": group["孕妇BMI"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def tree_summary(right_add):
    per = person_times(right_add)
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=60, random_state=42)
    tree.fit(per[["BMI"]], per["time"])
    per["leaf"] = tree.apply(per[["BMI"]])
    leaves = per.groupby("leaf")["BMI"].mean().sort_values().index
    mapping = {leaf: f"G{i+1}" for i, leaf in enumerate(leaves)}
    per["group"] = per["leaf"].map(mapping)
    thresholds = sorted(float(x) for x in tree.tree_.threshold if x > 0)
    groups = {}
    for name, group in per.groupby("group"):
        groups[name] = {
            "n": int(len(group)),
            "left": int((group.status == "left").sum()),
            "hit": int((group.status == "hit").sum()),
            "right": int((group.status == "right").sum()),
            "p90": float(np.percentile(group.time, 90)),
            "p95": float(np.percentile(group.time, 95)),
        }
    return {"thresholds": thresholds, "groups": groups}, per


right_sensitivity = {}
baseline_per = None
for offset in [0, 2, 4, 6, 8]:
    summary, current = tree_summary(offset)
    right_sensitivity[str(offset)] = summary
    if offset == 4:
        baseline_per = current
out["q2_right_censor_imputation_sensitivity"] = right_sensitivity
out["q2_status_total"] = baseline_per["status"].value_counts().to_dict()

# Female label consistency and repeated-person leakage audit.
label_nunique = female.groupby("孕妇代码")["异常"].nunique()
out["q4_label"] = {
    "abnormal_rows": int(female["异常"].sum()),
    "abnormal_women_any": int(female.groupby("孕妇代码")["异常"].max().sum()),
    "women_with_inconsistent_row_labels": int((label_nunique > 1).sum()),
}

features = [
    "X染色体的Z值",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体浓度",
    "GC含量",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "原始读段数",
    "唯一比对的读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "被过滤掉读段数的比例",
    "孕妇BMI",
    "年龄",
    "week",
]
f = female.dropna(subset=features).copy().reset_index(drop=True)
f["maxZ"] = f[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].max(axis=1)
f["minZ"] = f[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].min(axis=1)
f["log读段数"] = np.log10(f["唯一比对的读段数"] + 1)
f["Z极差"] = f["maxZ"] - f["minZ"]
features += ["maxZ", "minZ", "log读段数", "Z极差"]
X = f[features].to_numpy()
y = f["异常"].to_numpy()
groups = f["孕妇代码"].to_numpy()


def cv_oof(model, splitter, grouped):
    probabilities = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1)
    fold_metrics = []
    overlaps = []
    split_iter = splitter.split(X, y, groups if grouped else None)
    for train, test in split_iter:
        fitted = clone(model).fit(X[train], y[train])
        probabilities[test] = fitted.predict_proba(X[test])[:, 1]
        predictions[test] = fitted.predict(X[test])
        overlaps.append(len(set(groups[train]).intersection(set(groups[test]))))
        fold_metrics.append(
            {
                "auc": float(roc_auc_score(y[test], probabilities[test])),
                "ap": float(average_precision_score(y[test], probabilities[test])),
                "f1": float(f1_score(y[test], predictions[test], zero_division=0)),
            }
        )
    cm = confusion_matrix(y, predictions)
    return {
        "auc_oof": float(roc_auc_score(y, probabilities)),
        "ap_oof": float(average_precision_score(y, probabilities)),
        "f1_oof": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "sensitivity": float(cm[1, 1] / cm[1].sum()),
        "specificity": float(cm[0, 0] / cm[0].sum()),
        "fold_mean": {
            key: float(np.mean([row[key] for row in fold_metrics])) for key in ["auc", "ap", "f1"]
        },
        "fold_sd": {
            key: float(np.std([row[key] for row in fold_metrics])) for key in ["auc", "ap", "f1"]
        },
        "train_test_group_overlap_each_fold": overlaps,
    }


rf = RandomForestClassifier(
    n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
)
log_raw = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
log_scaled = make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
)
row_cv = StratifiedKFold(5, shuffle=True, random_state=42)
group_cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
out["q4_cv"] = {
    "row_stratified_rf": cv_oof(rf, row_cv, grouped=False),
    "group_stratified_rf": cv_oof(rf, group_cv, grouped=True),
    "row_stratified_scaled_logistic": cv_oof(log_scaled, row_cv, grouped=False),
    "group_stratified_raw_logistic": cv_oof(log_raw, group_cv, grouped=True),
    "group_stratified_scaled_logistic": cv_oof(log_scaled, group_cv, grouped=True),
}

# Compare Q3 timing propagation using residual SD versus directly estimable technical SD.
baseline_per = baseline_per.copy()
baseline_per["group"] = pd.cut(
    baseline_per["BMI"],
    bins=[-np.inf, 29.93, 31.57, 33.62, np.inf],
    labels=["G1", "G2", "G3", "G4"],
)
rng = np.random.default_rng(2025)


def timing_mc(sig, k=10000):
    result = {}
    slope = raw_mixed["week_beta"]
    for group_name, group in baseline_per.groupby("group", observed=True):
        eps = rng.normal(0, sig, (len(group), k))
        simulated = group["time"].to_numpy()[:, None] - eps / slope
        flat = simulated.ravel()
        result[str(group_name)] = {
            "q90": float(np.percentile(flat, 90)),
            "q95": float(np.percentile(flat, 95)),
            "p_reached_by_25": float(np.mean(flat <= 25)),
            "time_sd_from_error": float(sig / slope),
        }
    return result


out["q3_error_source_comparison"] = {
    "paper_residual_sd": timing_mc(raw_mixed["resid_sd"]),
    "technical_replicate_sd": timing_mc(tech_sd),
}

print(json.dumps(out, ensure_ascii=False, indent=2, default=float))
