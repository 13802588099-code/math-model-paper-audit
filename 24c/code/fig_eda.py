# -*- coding: utf-8 -*-
"""EDA 探索性数据分析图（不依赖求解结果）。"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, 'code')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotstyle as ps
import crops_data as cd
from matplotlib.patches import Patch

os.makedirs('figures', exist_ok=True)

# ---------- 图1：地块类型面积构成 ----------
def fig_land_structure():
    areas = {t: sum(cd.land_area[p] for p in ps_) for t, ps_ in cd.type_plots.items()}
    cols = ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']
    vals = [areas[c] for c in cols]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(cols, vals, color=ps.PALETTE[:6], edgecolor='white', height=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 8, b.get_y() + b.get_height()/2, f'{v:.1f}亩', va='center', fontsize=10)
    ax.set_xlabel('面积（亩）')
    ax.set_title('乡村各类耕地面积构成')
    ax.set_xlim(0, 700)
    ax.invert_yaxis()
    ps.save(fig, 'figures/fig_land_structure.png')

# ---------- 图2：2023年地块-作物种植格局热图 ----------
def fig_plot2023_heatmap():
    # 地块（按类型分组排序）x 年份? 这里是2023静态
    plots = []
    for t in ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']:
        plots += cd.type_plots[t]
    cids = []
    for p in plots:
        c = None
        for (yr, season) in cd.slots_of(p):
            if yr == 2023:
                break
        # 用2023种植数据
        sub = cd.plant23[(cd.plant23['地块'] == p)]
        if len(sub):
            c = sub['作物编号'].values[0]
        cids.append(c if c is not None else -1)
    cmap = plt.cm.tab20
    fig, ax = plt.subplots(figsize=(9, 7))
    # 用颜色代表作物
    allc = sorted(set(c for c in cids if c is not None))
    cmap_dict = {c: cmap(i % 20) for i, c in enumerate(allc)}
    ypos = range(len(plots))
    for i, p in enumerate(plots):
        c = cids[i]
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=cmap_dict.get(c, 'white'), ec='white'))
        ax.text(1.02, i, f'{p}({cd.land_area[p]:g}亩)', va='center', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(-1, len(plots) + 1)
    ax.axis('off')
    # 图例
    handles = [Patch(color=cmap_dict[c], label=cd.crop_id[c]) for c in sorted(allc)]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.25, 0.5), fontsize=8, ncol=2,
              title='2023年作物')
    ax.set_title('2023年各地块种植作物分布', fontsize=12)
    ps.save(fig, 'figures/fig_plot2023_heatmap.png', w=9, h=7)

# ---------- 图3：各类作物亩收益对比（横条） ----------
def fig_profit_compare():
    # 单季粮食地块（平旱地代表）+ 水稻 + 蔬菜(水浇地) + 食用菌(普通大棚)
    rows = []
    for p, c in [('A1', 1), ('A1', 2), ('A1', 3), ('A1', 4), ('A1', 5), ('A1', 6),
                 ('A1', 7), ('A1', 8), ('A1', 9), ('A1', 10), ('A1', 11), ('A1', 12),
                 ('A1', 13), ('A1', 14), ('A1', 15)]:
        rows.append((cd.crop_id[c], '粮食', cd.grain_pm(p, c)))
    rows.append(('水稻', '粮食', cd.pm('D1', 16, '第一季')))
    for c in cd.veg_first:
        if (c, '第一季') in cd.S0:
            rows.append((cd.crop_id[c], '蔬菜', cd.pm('D1', c, '第一季')))
    for c in cd.mushroom:
        rows.append((cd.crop_id[c], '食用菌', cd.pm('E1', c, '第二季')))
    df = pd.DataFrame(rows, columns=['作物', '类型', '亩收益'])
    df = df.sort_values('亩收益')
    fig, ax = plt.subplots(figsize=(7.6, 8))
    cmap = {'粮食': ps.PALETTE[0], '蔬菜': ps.PALETTE[1], '食用菌': ps.PALETTE[3]}
    cols = [cmap[t] for t in df['类型']]
    ax.barh(df['作物'], df['亩收益'], color=cols, edgecolor='white')
    ax.set_xlabel('每亩净收益（元/亩）')
    ax.set_title('2023年各作物每亩净收益（平旱地/水浇地/大棚口径）')
    ax.legend(handles=[Patch(color=cmap[k], label=k) for k in cmap], loc='lower right')
    ps.save(fig, 'figures/fig_profit_compare.png', w=7.6, h=8)

# ---------- 图4：2023产量（销售基准）条形图 ----------
def fig_sales_volume():
    S = cd.S0
    df = pd.DataFrame([(cd.crop_id[c], s, v) for (c, s), v in S.items()],
                      columns=['作物', '季次', '2023产量'])
    df['类型'] = df['作物'].map(cd.crop_type)
    df = df.sort_values('2023产量', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    cols = [{'粮食': ps.PALETTE[0], '蔬菜': ps.PALETTE[1], '食用菌': ps.PALETTE[3]}.get(t, '#999')
            for t in df['类型']]
    ax.barh(df['作物'] + '·' + df['季次'], df['2023产量'] / 1000, color=cols, edgecolor='white')
    ax.set_xlabel('2023年产量（千斤）')
    ax.set_title('各作物各季预期销售量基准（=2023年产量）')
    ps.save(fig, 'figures/fig_sales_volume.png', w=8, h=8)

if __name__ == '__main__':
    fig_land_structure()
    fig_plot2023_heatmap()
    fig_profit_compare()
    fig_sales_volume()
