# -*- coding: utf-8 -*-
"""汇总求解结果，输出论文所需的全部数值：
Q1两种情形总收益/结构占比/主要作物；Q2/Q3各λ风险指标；
Q1情形1 ±5%灵敏度表；各方案约束校验结果。
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
import numpy as np
sys.path.insert(0, 'code')
import crops_data as cd
from validate_plan import validate

YEARS = list(range(2024, 2031))
CATS = {'粮食(含豆类)': cd.grains + [cd.rice], '蔬菜': cd.veg_first + cd.veg_water_second,
        '食用菌': cd.mushroom}

def plant23_area(ids):
    """2023年各类作物种植面积(亩)。"""
    tot = 0.0
    for _, r in cd.plant23.iterrows():
        if int(r['作物编号']) in ids:
            tot += cd.land_area[r['地块']]
    return tot

def load_plant(tag):
    p = f'out/plan_{tag}.json'
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    plant = {}
    for k, c in d['plant'].items():
        yr, s, p_ = k.split('|'); plant[(int(yr), s, p_)] = c
    return plant, d

def area_share(plant):
    """各类作物7年累计种植面积占比与2023对比。"""
    share = {}
    for yr in YEARS:
        for cat, ids in CATS.items():
            a = sum(cd.land_area[p] for p in cd.plots_all for s in ['第一季', '第二季']
                    if plant.get((yr, s, p)) in ids)
            share.setdefault(cat, []).append(a)
    tot = np.array([sum(share[c][i] for c in CATS) for i in range(len(YEARS))])
    out = {c: [round(share[c][i] / tot[i], 4) for i in range(len(YEARS))] for c in CATS}
    return out

def crop_area_tot(plant, cids):
    """指定作物集合7年总种植面积(亩)。"""
    return sum(cd.land_area[p] for yr in YEARS for p in cd.plots_all
               for s in ['第一季', '第二季'] if plant.get((yr, s, p)) in cids)

def fixed_profit(plant, pf=1.0, yf=1.0, sf=1.0, cf=1.0, mode='waste'):
    tot = 0.0
    for (yr, season, p), c in plant.items():
        A = cd.land_area[p]; pr = cd.unit_price(c, season) * pf
        co = cd.unit_cost(p, c, season) * cf * A
        prod = A * cd.unit_yield(p, c, season) * yf
        S = cd.S0[(c, season)] * sf
        u = min(prod, S)
        tot += pr * u - co
        if mode == 'discount' and prod > S:
            tot += 0.5 * pr * (prod - S)
    return tot

print('=' * 60)
print('问题1')
print('=' * 60)
for tag, mode, lab in [('q1_1', 'waste', '情形1(超产滞销)'), ('q1_2', 'discount', '情形2(超产50%折价)')]:
    plant, d = load_plant(tag)
    if d is None:
        print(f'  [{tag}] 缺结果'); continue
    print(f'   {lab}: 总收益 = {d["profit"]:,.0f} 元 = {d["profit"]/1e4:,.1f} 万元')
    sh = area_share(plant)
    print('    各类面积占比 2024→2030:', {c: sh[c][0] for c in CATS}, '→', {c: sh[c][-1] for c in CATS})
    # 2023 基准
    sh23 = {cat: plant23_area(ids) for cat, ids in CATS.items()}
    tot23 = sum(sh23.values())
    if tot23 > 0:
        print('    2023占比:', {c: round(sh23[c] / tot23, 3) for c in CATS})
    ok, iss = validate(plant, mode=mode)
    print(f'    约束校验: {"✓通过" if ok else f"{len(iss)}项"}')
    # 主要作物面积Top
    top = sorted({c for (yr, s, p), c in plant.items()},
                 key=lambda c: crop_area_tot(plant, [c]), reverse=True)[:6]
    print('    种植面积Top作物:', [(cd.crop_id[c], round(crop_area_tot(plant, [c]))) for c in top])

print()
print('=' * 60)
print('问题2/3 风险指标 (λ 扫描)')
print('=' * 60)
for tag in ['q2', 'q3']:
    for lam in ([0.0, 0.3, 0.8] if tag == 'q2' else [0.3, 0.8]):
        p = f'out/plan_{tag}_l{lam}.json'
        if not os.path.exists(p):
            print(f'  [{tag} λ={lam}] 缺结果'); continue
        d = json.load(open(p))
        print(f'   [{tag} λ={lam}] E={d["E_profit"]:,.0f}元({d["E_profit"]/1e4:,.1f}万) '
              f'std={d["std"]:,.0f} 最差={d["worst"]:,.0f} CVaR损失={d["cvar"]:,.0f} '
              f'亏损概率={d["loss_prob"]:.3f}')

print()
print('=' * 60)
print('Q1情形1 ±5% 灵敏度表（固定方案重算）')
print('=' * 60)
plant1, d1 = load_plant('q1_1')
if plant1:
    base = fixed_profit(plant1, mode='waste')
    for name, kw in [('销售单价 r', 'pf'), ('亩产量 y', 'yf'), ('销售量 S', 'sf'), ('种植成本 c', 'cf')]:
        lo = fixed_profit(plant1, mode='waste', **{kw: 0.95})
        hi = fixed_profit(plant1, mode='waste', **{kw: 1.05})
        span = (hi - lo) / 1e4
        print(f'   {name} ±5%: 收益变化 -{ (base-lo)/1e4:,.1f} ~ +{ (hi-base)/1e4:,.1f} 万元, 跨度 {span:,.1f} 万元')

# 2023年产量/收益基准
def plant23_get(cdmod, p, s, ids):
    for _, r in cdmod.plant23.iterrows():
        if r['地块'] == p and r['季次'] == s and int(r['作物编号']) in ids:
            return True
    return False

print()
print('2023年总产量(斤)与估算收益:')
tot23_prod = sum(cd.S0.values())
print(f'  2023总产量 = {tot23_prod:,.0f} 斤')
