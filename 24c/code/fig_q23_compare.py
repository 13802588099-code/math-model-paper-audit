# -*- coding: utf-8 -*-
"""问题2 vs 问题3 风险对比图：情景收益分布对比、风险指标对比、λ敏感性。
突出"考虑作物间相关性后，极端情景风险显著放大"这一核心结论。
"""
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

def load(tag, lam='0.3'):
    d = json.load(open(f'out/plan_{tag}_l{lam}.json'))
    p = np.array(d['per_scenario_profit'])
    return d, p

def risk(p, alpha=0.90):
    n = len(p); nt = max(1, int(np.ceil((1 - alpha) * n)))
    E = p.mean(); sd = p.std(); worst = p.min()
    cvar = -np.sort(p)[:nt].mean()          # 损失 = -profit
    return E, sd, worst, cvar

# ---------- 图1：Q2 vs Q3 情景收益分布（相关性放大尾部风险） ----------
def fig_dist_compare():
    _, p2 = load('q2'); _, p3 = load('q3')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(p2/1e4, ax=ax, fill=True, alpha=0.25, color=ps.PALETTE[0], label='问题2（独立情景）')
    sns.kdeplot(p3/1e4, ax=ax, fill=True, alpha=0.25, color=ps.PALETTE[2], label='问题3（相关情景）')
    for data, c in [(p2, ps.PALETTE[0]), (p3, ps.PALETTE[2])]:
        ax.axvline(data.mean()/1e4, color=c, ls='--', lw=1.2)
        ax.axvline(data.min()/1e4, color=c, ls=':', lw=1.2)
    ax.set_xlabel('2024—2030 总净收益（万元）')
    ax.set_ylabel('概率密度')
    ax.set_title('问题2与问题3情景收益分布对比')
    ax.legend()
    ps.save(fig, 'figures/fig_q23_dist.png', w=8, h=5)

# ---------- 图2：Q2 vs Q3 风险指标对比 ----------
def fig_risk_compare():
    E2, sd2, w2, cv2 = risk(load('q2')[1])
    E3, sd3, w3, cv3 = risk(load('q3')[1])
    rows = [
        ('标准差(万元)',  sd2/1e4,  sd3/1e4),
        ('最差偏离均值(%)', 100*(1 - w2/E2), 100*(1 - w3/E3)),
        ('CVaR短缺口(万元)', (E2 + cv2)/1e4, (E3 + cv3)/1e4),
        ('最差情景收益(万元)', w2/1e4, w3/1e4),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # 左：三根风险条
    names = [r[0] for r in rows[:3]]; v2 = [r[1] for r in rows[:3]]; v3 = [r[2] for r in rows[:3]]
    x = np.arange(len(names)); w = 0.34
    for i, v in enumerate(v2):
        axes[0].bar(x[i] - w/2, v, w, color=ps.PALETTE[0], label='问题2' if i == 0 else None)
    for i, v in enumerate(v3):
        axes[0].bar(x[i] + w/2, v, w, color=ps.PALETTE[2], label='问题3' if i == 0 else None)
    for i in range(len(names)):
        axes[0].text(x[i] - w/2, v2[i] + max(v2+v3)*0.02, f'{v2[i]:.1f}', ha='center', fontsize=9)
        axes[0].text(x[i] + w/2, v3[i] + max(v2+v3)*0.02, f'{v3[i]:.1f}', ha='center', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel('数值'); axes[0].legend()
    axes[0].set_title('相关性放大组合风险')
    # 右：最差情景收益对比
    names2 = ['问题2', '问题3']
    y = [w2/1e4, w3/1e4]
    bars = axes[1].bar(names2, y, 0.5, color=[ps.PALETTE[0], ps.PALETTE[2]])
    for b, v in zip(bars, y):
        axes[1].text(b.get_x() + b.get_width()/2, v + 5, f'{v:,.0f}', ha='center')
    axes[1].set_ylabel('最差情景收益（万元）')
    axes[1].set_title('最差情景收益')
    fig.tight_layout()
    ps.save(fig, 'figures/fig_q23_risk.png', w=11, h=4.4)

# ---------- 图3：λ敏感性（Q2/Q3 的 E 与 CVaR 随 λ 变化） ----------
def fig_lambda_sens():
    lams = [0.0, 0.3, 0.6, 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for tag, c, lab in [('q2', ps.PALETTE[0], '问题2'), ('q3', ps.PALETTE[2], '问题3')]:
        E, CV = [], []
        for lam in lams:
            if not os.path.exists(f'out/plan_{tag}_l{lam}.json'):
                continue
            p = np.array(json.load(open(f'out/plan_{tag}_l{lam}.json'))['per_scenario_profit'])
            e_, sd_, w_, cv_ = risk(p)
            E.append(e_/1e4); CV.append((e_ + cv_)/1e4)   # CVaR 短缺口 = E - CVaR
        axes[0].plot(lams[:len(E)], E, '-o', color=c, label=lab)
        axes[1].plot(lams[:len(CV)], CV, '-s', color=c, label=lab)
    axes[0].set_xlabel('风险规避系数 λ'); axes[0].set_ylabel('期望收益（万元）')
    axes[0].set_title('期望收益对 λ 的敏感性'); axes[0].legend()
    axes[1].set_xlabel('风险规避系数 λ'); axes[1].set_ylabel('CVaR短缺口（万元）')
    axes[1].set_title('CVaR短缺口对 λ 的敏感性'); axes[1].legend()
    fig.tight_layout()
    ps.save(fig, 'figures/fig_q23_lambda.png', w=11, h=4.2)

if __name__ == '__main__':
    fig_dist_compare()
    fig_risk_compare()
    fig_lambda_sens()
    print('✓ fig_q23_compare 完成')
