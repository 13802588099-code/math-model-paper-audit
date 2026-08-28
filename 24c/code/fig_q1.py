# -*- coding: utf-8 -*-
"""问题1方案可视化：热图、逐年面积堆叠、产量vs销售、情形对比。"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'code')
import plotstyle as ps
import crops_data as cd
from matplotlib.patches import Patch

os.makedirs('figures', exist_ok=True)

def load_plan(tag):
    d = json.load(open(f'out/plan_{tag}.json'))
    plant = {}
    for k, c in d['plant'].items():
        yr, s, p = k.split('|')
        plant[(int(yr), s, p)] = c
    return plant, d

# ---------- 图A：7年种植方案热图（地块×年份，颜色=作物类型） ----------
def fig_plan_heatmap(plant, tag, fname):
    plots = []
    for t in ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']:
        plots += cd.type_plots[t]
    years = list(range(2024, 2031))
    cmap = {1: '#F2C94C', 2: '#F2994A', 3: '#EB5757', 4: '#9B51E0', 5: '#2D9CDB',
            6: '#27AE60', 7: '#6FCF97', 8: '#BB6BD9', 9: '#56CCF2', 10: '#219653',
            11: '#F2994A', 12: '#C4E538', 13: '#F6C445', 14: '#E55039', 15: '#0C5C8E',
            16: '#7ED6DF', 17: '#4B6584', 18: '#B33771', 19: '#6D214F', 20: '#F8B400',
            21: '#EA2027', 22: '#009432', 23: '#1289A7', 24: '#D980FA', 25: '#FD7272',
            26: '#A3CB38', 27: '#B53471', 28: '#006266', 29: '#FDA7DF', 30: '#ED4C67',
            31: '#F8C291', 32: '#82CCDD', 33: '#D6A2E8', 34: '#B8E994', 35: '#78E08F',
            36: '#FAD7A0', 37: '#E59866', 38: '#5499C7', 39: '#AF7AC5', 40: '#48C9B0',
            41: '#EC7063'}
    fig, ax = plt.subplots(figsize=(12, 8))
    for yi, yr in enumerate(years):
        for pi, p in enumerate(plots):
            c = plant.get((yr, '第一季', p))
            if c is None:
                c = plant.get((yr, '第二季', p))
            col = cmap.get(c, 'white') if c is not None else 'white'
            ax.add_patch(plt.Rectangle((yi, pi), 1, 1, color=col, ec='0.85', lw=0.3))
    ax.set_xlim(0, len(years)); ax.set_ylim(-1, len(plots) + 1)
    ax.set_xticks([i + 0.5 for i in range(len(years))])
    ax.set_xticklabels([f'{y}' for y in years])
    ax.set_yticks([i + 0.5 for i in range(len(plots))])
    ax.set_yticklabels([f'{p}' for p in plots], fontsize=6)
    ax.set_xlabel('年份'); ax.set_ylabel('地块')
    ax.set_title('2024-2030年种植方案（颜色=作物，空白=休耕/未种）')
    # 图例
    used = sorted({plant.get((yr, s, p)) for yr in years for s in ['第一季', '第二季']
                   for p in plots} - {None})
    handles = [Patch(color=cmap.get(c, '#ccc'), label=cd.crop_id[c]) for c in used]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=7, fontsize=7, frameon=False)
    ps.save(fig, f'figures/{fname}', w=12, h=8)

# ---------- 图B：各作物逐年种植面积堆叠图 ----------
def fig_area_stack(plant, tag, fname):
    years = list(range(2024, 2031))
    # 计算每年各作物面积（第一季+第二季合并，但按作物）
    area_df = {c: [] for c in cd.crop_ids_all}
    for yr in years:
        for c in cd.crop_ids_all:
            a = 0.0
            for p in cd.plots_all:
                for s in ['第一季', '第二季']:
                    if plant.get((yr, s, p)) == c:
                        a += cd.land_area[p]
            area_df[c].append(a)
    df = pd.DataFrame(area_df, index=years).T
    # 合并小类别
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    df.T.plot.area(ax=ax, colormap='tab20', linewidth=0)
    ax.set_xlabel('年份'); ax.set_ylabel('种植面积（亩）')
    ax.set_title('各作物种植面积逐年变化')
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ps.save(fig, f'figures/{fname}', w=8.5, h=5.5)

# ---------- 图C：情形1 vs 情形2 总收益对比 ----------
def fig_cases_compare(d1, d2):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    vals = [d1['profit'] / 1e4, d2['profit'] / 1e4]
    labels = ['情形1\n(超产滞销)', '情形2\n(超产50%降价)']
    bars = ax.bar(labels, vals, color=[ps.PALETTE[0], ps.PALETTE[1]], width=0.5, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f'{v:.0f}万元', ha='center', fontsize=11)
    ax.set_ylabel('2024-2030年总净收益（万元）')
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title('问题1两种情形总收益对比')
    ps.save(fig, 'figures/fig_cases_compare.png', w=6, h=4.2)

# ---------- 图D：各作物产量 vs 销售量（情形1，2030年为例） ----------
def fig_prod_vs_sales(plant, tag, fname, yr=2030, logx=False):
    # 计算该作物当年产量总和 vs 销售基准
    rows = []
    for (c, season) in cd.crop_season_pairs:
        prod = 0.0
        for p in cd.plots_all:
            if plant.get((yr, season, p)) == c:
                prod += cd.land_area[p] * cd.unit_yield(p, c, season)
        S = cd.S0[(c, season)]
        rows.append((cd.crop_id[c], season, prod / 1000, S / 1000))
    df = pd.DataFrame(rows, columns=['作物', '季次', '产量', '销售量'])
    df = df.sort_values('产量')
    fig, ax = plt.subplots(figsize=(8, 8))
    x = np.arange(len(df)); w = 0.38
    ax.barh(x - w/2, df['产量'], w, label='产量', color=ps.PALETTE[0])
    ax.barh(x + w/2, df['销售量'], w, label='销售量基准', color=ps.PALETTE[4])
    ax.set_yticks(x); ax.set_yticklabels(df['作物'] + '·' + df['季次'], fontsize=7)
    ax.set_xlabel('千斤' + ('（对数轴）' if logx else ''))
    if logx:
        ax.set_xscale('log')
    ax.set_title(f'{yr}年 各作物产量与预期销售量对比（{tag}）')
    ax.legend()
    ps.save(fig, f'figures/{fname}', w=8, h=8)

if __name__ == '__main__':
    p1, d1 = load_plan('q1_1')
    p2, d2 = load_plan('q1_2')
    fig_plan_heatmap(p1, 'q1_1', 'fig_plan_q1_1.png')
    fig_plan_heatmap(p2, 'q1_2', 'fig_plan_q1_2.png')
    fig_area_stack(p1, 'q1_1', 'fig_area_q1_1.png')
    fig_area_stack(p2, 'q1_2', 'fig_area_q1_2.png')
    fig_cases_compare(d1, d2)
    fig_prod_vs_sales(p1, '情形1', 'fig_prod_vs_sales_q1_1.png')
    fig_prod_vs_sales(p2, '情形2', 'fig_prod_vs_sales_q1_2.png', logx=True)
