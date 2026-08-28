# -*- coding: utf-8 -*-
"""问题2/3 风险分析图：情景收益分布、收益-风险前沿、风险指标对比、种植结构变化。"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, 'code')
import plotstyle as ps
import crops_data as cd

sns.set_theme(style='whitegrid')
os.makedirs('figures', exist_ok=True)

def load_meta(tag, lambdas):
    """读取各 λ 方案的指标与情景收益。"""
    out = []
    for lam in lambdas:
        p = f'out/plan_{tag}_l{lam}.json'
        if not os.path.exists(p):
            print(f'  [跳过] 缺少 {p}')
            continue
        d = json.load(open(p))
        out.append((lam, d))
    return out

# ---------- 图A：不同 λ 下情景收益分布 ----------
def fig_scen_dist(tag, lambdas, fname, taglabel):
    data = load_meta(tag, lambdas)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (lam, d) in enumerate(data):
        prof = np.array(d['per_scenario_profit']) / 1e4
        sns.kdeplot(prof, ax=ax, label=f'λ={lam}', fill=True, alpha=0.25,
                    color=ps.PALETTE[i])
        ax.axvline(prof.mean(), color=ps.PALETTE[i], ls='--', lw=1)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('总净收益（万元）')
    ax.set_ylabel('概率密度')
    ax.set_title(f'{taglabel}：不同风险偏好下情景收益分布')
    ax.legend(title='风险系数')
    ps.save(fig, f'figures/{fname}', w=8, h=5)

# ---------- 图B：收益-风险前沿（期望收益 vs CVaR/标准差） ----------
def fig_frontier(tag, lambdas, fname, taglabel):
    data = load_meta(tag, lambdas)
    if len(data) < 2:
        return
    E = np.array([d['E_profit'] for _, d in data]) / 1e4
    cvar = np.array([d['cvar'] for _, d in data]) / 1e4
    std = np.array([d['std'] for _, d in data]) / 1e4
    lams = [lam for lam, _ in data]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cvar, E, '-o', color=ps.PALETTE[0], lw=2, ms=7, label='期望收益—CVaR')
    for x, y, lam in zip(cvar, E, lams):
        ax.annotate(f'λ={lam}', (x, y), textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax2 = ax.twiny()
    ax2.plot(std, E, '--s', color=ps.PALETTE[3], lw=1.5, ms=6, label='期望收益—标准差')
    ax.set_xlabel('CVaR(损失)（万元）'); ax.set_ylabel('期望收益（万元）')
    ax2.set_xlabel('收益标准差（万元）')
    ax.set_title(f'{taglabel}：收益-风险前沿')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower right')
    ps.save(fig, f'figures/{fname}', w=7, h=5)

# ---------- 图C：各 λ 风险指标分组对比 ----------
def fig_risk_metrics(tag, lambdas, fname, taglabel):
    data = load_meta(tag, lambdas)
    if not data:
        return
    df = pd.DataFrame([{'λ': lam, '期望收益': d['E_profit']/1e4, '最差情景': d['worst']/1e4,
                        'CVaR损失': d['cvar']/1e4, '标准差': d['std']/1e4}
                       for lam, d in data])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cols = ['期望收益', '最差情景', 'CVaR损失', '标准差']
    x = np.arange(len(df)); w = 0.2
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, df[col], w, label=col, color=ps.PALETTE[i])
    ax.set_xticks(x); ax.set_xticklabels([f'λ={l}' for l in df['λ']])
    ax.set_ylabel('金额（万元）')
    ax.set_title(f'{taglabel}：不同风险偏好下风险指标对比')
    ax.legend(ncol=2)
    ps.save(fig, f'figures/{fname}', w=7.5, h=4.6)

# ---------- 图D：不同 λ 下各作物种植面积占比变化 ----------
def fig_area_by_lambda(tag, lambdas, fname, taglabel):
    data = load_meta(tag, lambdas)
    if not data:
        return
    years = list(range(2024, 2031))
    # 计算每个 λ 下 7 年总种植面积（按作物类别）
    cats = {'粮食': cd.grains + [cd.rice], '蔬菜': cd.veg_first,
            '水浇地二季菜': cd.veg_water_second, '食用菌': cd.mushroom}
    rows = []
    for lam, d in data:
        plant = {}
        for k, c in d['plant'].items():
            yr, s, p = k.split('|')
            plant[(int(yr), s, p)] = c
        for cat, ids in cats.items():
            area = 0.0
            for yr in years:
                for p in cd.plots_all:
                    for s in ['第一季', '第二季']:
                        c = plant.get((yr, s, p))
                        if c in ids:
                            area += cd.land_area[p]
            rows.append({'λ': f'λ={lam}', '类别': cat, '种植面积（亩）': area})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x='λ', y='种植面积（亩）', hue='类别', ax=ax,
                palette=[ps.PALETTE[0], ps.PALETTE[2], ps.PALETTE[4], ps.PALETTE[7]])
    ax.set_title(f'{taglabel}：不同风险偏好下各类作物种植面积')
    ax.legend(title='作物类别', ncol=2)
    ps.save(fig, f'figures/{fname}', w=8, h=5)

if __name__ == '__main__':
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else 'q2'
    lambdas = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.0, 0.3, 0.8]
    label = '问题2' if tag == 'q2' else '问题3'
    fig_scen_dist(tag, lambdas, f'fig_scen_dist_{tag}.png', label)
    fig_frontier(tag, lambdas, f'fig_frontier_{tag}.png', label)
    fig_risk_metrics(tag, lambdas, f'fig_risk_metrics_{tag}.png', label)
    fig_area_by_lambda(tag, lambdas, f'fig_area_{tag}.png', label)
