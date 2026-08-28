# -*- coding: utf-8 -*-
"""NIPT C题 公共模块：数据加载、孕周解析、全局绘图样式。"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

DATA_FILE = "/Users/lzt778899/2026数模国赛/2025赛题/SvpohSGacdffe718bcaa3b6e835c03ae3461cab1/C题/附件.xlsx"
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "texfile", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------- 中文字体
_CANDIDATES = ["SimSun", "Songti SC", "STSong", "Heiti SC", "Hiragino Sans GB"]
_AVAIL = {f.name for f in fm.fontManager.ttflist}
_FONT = next((c for c in _CANDIDATES if c in _AVAIL), "Songti SC")


def setup_style():
    """统一全局绘图样式。"""
    plt.rcParams.update({
        "font.sans-serif": [_FONT, "SimSun", "Songti SC", "STSong", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    sns.set_theme(style="whitegrid", context="notebook",
                  palette="deep", rc={"font.sans-serif": [_FONT, "Arial"],
                                      "axes.unicode_minus": False})


# 统一调色板（蓝—青—橙—红—紫）
PALETTE = ["#2c6fbb", "#20b2aa", "#e8853a", "#d94f4f", "#8e6bb2",
           "#5b8f5b", "#c49a3a", "#6b6b6b"]


def save(fig, name):
    """保存图形到 texfile/figures。"""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[saved]", path)
    return path


def parse_week(s):
    """'11w+6' -> 11.857；'23w' -> 23.0。"""
    s = str(s).strip().replace("周", "")
    m = re.match(r"(\d+)\s*w\s*\+?\s*(\d+)?", s)
    if not m:
        return np.nan
    w = int(m.group(1))
    d = int(m.group(2) or 0)
    return round(w + d / 7.0, 6)


def load_data():
    """加载并预处理男胎/女胎数据，返回 (male, female)。"""
    male = pd.read_excel(DATA_FILE, sheet_name="男胎检测数据")
    female = pd.read_excel(DATA_FILE, sheet_name="女胎检测数据")

    # 规范化列名（去空格）
    male.columns = [str(c).strip() for c in male.columns]
    female.columns = [str(c).strip() for c in female.columns]

    for df in (male, female):
        df["week"] = df["检测孕周"].apply(parse_week)
        # 衍生特征
        df["异常"] = df["染色体的非整倍体"].notna().astype(int)
    return male, female


def first_pass_table(df):
    """按孕妇代码聚合：第一次 Y 浓度 >= 0.04 的孕周（达标时间）。"""
    rows = []
    for code, g in df.groupby("孕妇代码"):
        g = g.sort_values("week")
        hit = g[g["Y染色体浓度"] >= 0.04]
        rows.append({
            "孕妇代码": code,
            "first_pass_week": hit["week"].min() if len(hit) else np.nan,
            "n_draws": len(g),
            "BMI": g["孕妇BMI"].iloc[0],
            "age": g["年龄"].iloc[0],
            "height": g["身高"].iloc[0],
            "weight": g["体重"].iloc[0],
            "IVF妊娠": g["IVF妊娠"].iloc[0],
            "健康": g["胎儿是否健康"].iloc[0],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    setup_style()
    male, female = load_data()
    print("male:", male.shape, "female:", female.shape)
    print("男胎孕妇数:", male["孕妇代码"].nunique())
    print("女胎孕妇数:", female["孕妇代码"].nunique())
    ft = first_pass_table(male)
    print("达标时间表:", ft.shape, "未达标人数:", ft["first_pass_week"].isna().sum())
