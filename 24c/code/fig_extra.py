# -*- coding: utf-8 -*-
"""补充高级图表：灵敏度龙卷风图、风险指标雷达图、情景收益小提琴图、
年度各类作物面积占比堆叠、产量聚类热图。
依赖 out/plan_*.json（不存在则跳过对应图）。
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'code')
import plotstyle as ps
import crops_data as cd

os.makedirs('figures', exist_ok=True)
YEARS = list(range(2024, 2031))
CATS = {'粮食(含豆类)': cd.grains + [cd.rice], '第一季蔬菜': cd.veg_first,
        '水浇地二季菜': cd.veg_water_second, '食用菌': cd.mushroom}


def load_plan(tag):
    d = json.load(open(f'out/plan_{tag}.json'))
    plant = {}
    for k, c in d['plant'].items():
        yr, s, p = k.split('|')
        plant[(int(yr), s, p)] = c
    return plant, d


# ---------- 固定方案下的收益核算（灵敏度分析用） ----------
def fixed_plan_profit(plant, mode='waste', pf=1.0, yf=1.0, sf=1.0, cf=1.0):
    """按固定种植方案重算总收益。pf价格乘子/yf产量乘子/sf销售量乘子/cf成本乘子。"""
    tot = 0.0
    for (yr, season, p), c in plant.items():
        A = cd.land_area[p]
        pr = cd.unit_price(c, season) * pf
        co = cd.unit_cost(p, c, season) * cf * A
        prod = A * cd.unit_yield(p, c, season) * yf
        S = cd.S0[(c, season)] * sf
        u = min(prod, S)
        tot += pr * u - co
        if mode == 'discount' and prod > S:
            tot += 0.5 * pr * (prod - S)
    return tot


# ---------- 图E：灵敏度龙卷风图（Q1情形1） ----------
def fig_tornado(tag, fname, mode='waste'):
    p = f'out/plan_{tag}.json'
    if not os.path.exists(p):
        print('  [跳过] 缺', p); return
    plant, d = load_plan(tag)
    base = fixed_plan_profit(plant, mode=mode)
    factors = [('价格', 'pf'), ('亩产量', 'yf'), ('销售量', 'sf'), ('成本', 'cf')]
    rows = []
    for name, key in factors:
        lo = fixed_plan_profit(plant, mode=mode, **{key: 0.9})
        hi = fixed_plan_profit(plant, mode=mode, **{key: 1.1})
        rows.append((name, (lo - base) / 1e4, (hi - base) / 1e4))
    df = pd.DataFrame(rows, columns=['因素', '下降10%', '上升10%']).sort_values('下降10%')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(df))
    ax.barh(y, df['上升10%'], left=0, color=ps.PALETTE[2], height=0.5, label='参数 +10%')
    ax.barh(y, df['下降10%'], left=0, color=ps.PALETTE[1], height=0.5, label='参数 −10%')
    ax.axvline(0, color='k', lw=1)
    ax.set_yticks(y); ax.set_yticklabels(df['因素'])
    ax.set_xlabel('总收益变化（万元）')
    ax.set_title('关键参数±10%对总收益的灵敏度（龙卷风图）')
    ax.legend(loc='lower right')
    ps.save(fig, f'figures/{fname}', w=7, h=3.8)


# ---------- 图F：风险指标雷达图（Q2 vs Q3，不同λ） ----------
def fig_radar(tags, lambdas, fname):
    data = []
    for tag in tags:
        for lam in lambdas:
            p = f'out/plan_{tag}_l{lam}.json'
            if os.path.exists(p):
                data.append((tag, lam, json.load(open(p))))
    if not data:
        print('  [跳过] 无雷达图数据'); return
    # 指标：期望收益、最差情景、CVaR损失、标准差 —— 均取万元
    metrics = ['E_profit', 'worst', 'cvar', 'std']
    lbl = {'E_profit': '期望收益', 'worst': '最差情景', 'cvar': 'CVaR损失', 'std': '标准差'}
    # min-max 归一（损失/标准差越小越好 → 取负向）
    df = pd.DataFrame([{'tag': t, 'λ': lam,
                        **{m: d[m] / 1e4 for m in metrics}} for t, lam, d in data])
    norm = pd.DataFrame(index=df.index)
    for m in metrics:
        lo, hi = df[m].min(), df[m].max()
        rng = (hi - lo) or 1.0
        norm[m] = (df[m] - lo) / rng
        if m in ('cvar', 'std'):
            norm[m] = 1 - norm[m]
    fig, ax = plt.subplots(figsize=(6.5, 5), subplot_kw={'projection': 'polar'})
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for i, (t, lam, d) in enumerate(data):
        vals = norm.loc[i, metrics].tolist(); vals += vals[:1]
        lab = f'{t} λ={lam}'
        ax.plot(angles, vals, '-o', lw=2, label=lab, color=ps.PALETTE[i % 10])
        ax.fill(angles, vals, alpha=0.08, color=ps.PALETTE[i % 10])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([lbl[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title('不同风险偏好下风险指标雷达图（归一化）', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=8)
    ps.save(fig, f'figures/{fname}', w=6.5, h=5)


# ---------- 图G：情景收益小提琴图 ----------
def fig_violin(tag, lambdas, fname, taglabel):
    data = []
    for lam in lambdas:
        p = f'out/plan_{tag}_l{lam}.json'
        if os.path.exists(p):
            d = json.load(open(p))
            data.append((lam, np.array(d['per_scenario_profit']) / 1e4))
    if not data:
        print('  [跳过] 缺', tag); return
    parts = []
    for lam, prof in data:
        parts.append(pd.DataFrame({'λ': f'λ={lam}', '收益（万元）': prof}))
    df = pd.concat(parts)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(data=df, x='λ', y='收益（万元）', ax=ax, inner='quart',
                   palette=[ps.PALETTE[0], ps.PALETTE[1], ps.PALETTE[2]][:len(data)])
    sns.pointplot(data=df, x='λ', y='收益（万元）', ax=ax, color='k', errorbar=None,
                  marker='D', scale=0.7)
    ax.axhline(0, color='0.6', ls='--', lw=0.8)
    ax.set_title(f'{taglabel}：不同λ下情景收益分布（小提琴图）')
    ps.save(fig, f'figures/{fname}', w=7, h=4.5)


# ---------- 图H：年度各类作物面积占比堆叠 ----------
def fig_area_share(tag, fname, taglabel):
    p = f'out/plan_{tag}.json'
    if not os.path.exists(p):
        print('  [跳过] 缺', p); return
    plant, d = load_plan(tag)
    share = {cat: [] for cat in CATS}
    for yr in YEARS:
        tot = 0.0
        per = {}
        for cat, ids in CATS.items():
            a = 0.0
            for p_ in cd.plots_all:
                for s in ['第一季', '第二季']:
                    if plant.get((yr, s, p_)) in ids:
                        a += cd.land_area[p_]
            per[cat] = a; tot += a
        for cat in CATS:
            share[cat].append(per[cat] / tot if tot else 0.0)
    df = pd.DataFrame(share, index=YEARS)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    df.plot.area(ax=ax, color=[ps.PALETTE[0], ps.PALETTE[1], ps.PALETTE[3], ps.PALETTE[4]],
                 linewidth=0)
    ax.set_xlabel('年份'); ax.set_ylabel('种植面积占比')
    ax.set_title(f'{taglabel}：各类作物种植面积占比逐年变化')
    ax.legend(title='作物类别', ncol=2, fontsize=8)
    ps.save(fig, f'figures/{fname}', w=8, h=4.6)


# ---------- 图I：产量聚类热图（作物×年份） ----------
def fig_clustermap(tag, fname, taglabel):
    p = f'out/plan_{tag}.json'
    if not os.path.exists(p):
        print('  [跳过] 缺', p); return
    plant, d = load_plan(tag)
    mat = np.zeros((len(cd.crop_season_pairs), len(YEARS)))
    for (c, season) in cd.crop_season_pairs:
        for yr in YEARS:
            prod = 0.0
            for p_ in cd.plots_all:
                if plant.get((yr, season, p_)) == c:
                    prod += cd.land_area[p_] * cd.unit_yield(p_, c, season)
            mat[cd.crop_season_pairs.index((c, season)), YEARS.index(yr)] = prod / 1e4
    rows = [f'{cd.crop_id[c]}·{s}' for (c, s) in cd.crop_season_pairs]
    df = pd.DataFrame(mat, index=rows, columns=[str(y) for y in YEARS])
    df = df[(df != 0).any(axis=1)]          # 只保留种植过的作物
    if df.empty:
        print('  [跳过] 聚类热图无数据'); return
    try:
        g = sns.clustermap(df, cmap='YlGnBu', linewidths=0.3, figsize=(9, 10),
                           yticklabels=True, xticklabels=True,
                           cbar_kws={'label': '产量（万斤）'})
        g.ax_heatmap.set_xlabel('年份'); g.ax_heatmap.set_ylabel('作物·季次')
        g.fig.suptitle(f'{taglabel}：作物产量年际聚类热图', y=1.02, fontsize=13)
        g.savefig(f'figures/{fname}', dpi=300, bbox_inches='tight')
        print('  ✓', f'figures/{fname}')
        plt.close(g.fig)
    except Exception as e:
        print('  [跳过] 聚类热图失败:', e)


if __name__ == '__main__':
    # Q1 两类灵敏度 + 占比堆叠 + 聚类热图
    fig_tornado('q1_1', 'fig_tornado_q1_1.png', mode='waste')
    fig_tornado('q1_2', 'fig_tornado_q1_2.png', mode='discount')
    fig_area_share('q1_1', 'fig_area_share_q1_1.png', '情形1')
    fig_area_share('q1_2', 'fig_area_share_q1_2.png', '情形2')
    fig_clustermap('q1_1', 'fig_clustermap_q1_1.png', '情形1')
    fig_clustermap('q1_2', 'fig_clustermap_q1_2.png', '情形2')
    # Q2/Q3 雷达 + 小提琴
    fig_radar(['q2'], [0.0, 0.3, 0.8], 'fig_radar_q2.png')
    fig_radar(['q3'], [0.3, 0.8], 'fig_radar_q3.png')
    fig_radar(['q2', 'q3'], [0.0, 0.3, 0.8], 'fig_radar_all.png')
    fig_violin('q2', [0.0, 0.3, 0.8], 'fig_violin_q2.png', '问题2')
    fig_violin('q3', [0.3, 0.8], 'fig_violin_q3.png', '问题3')
