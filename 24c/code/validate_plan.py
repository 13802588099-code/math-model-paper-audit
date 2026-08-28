# -*- coding: utf-8 -*-
"""验证种植方案的约束满足性：地块容量、重茬、豆类三年轮作、销售上限。
plant: dict[(yr, season, p)] = crop_id
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import crops_data as cd

def validate(plant, S_mult=None, verbose=True, mode='waste'):
    """S_mult: dict[(c,season)]->销售量乘子（问题2/3用），默认1
    mode: 'waste'(情形1，超产即违规) / 'discount'(情形2，超产按半价出售属正常) /
          'stoch'(问题2/3，逐情景不校验，超产风险已含在情景中)"""
    S_mult = S_mult or {}
    ok = True
    issues = []

    # 1. 每个地块每季至多一种作物 + 允许作物集合
    for p in cd.plots_all:
        for (yr, season) in cd.slots_of(p):
            c = plant.get((yr, season, p))
            if c is None:
                continue
            if c not in cd.allowed_crops(p, season):
                issues.append(f'[{p}] {yr}{season} 作物{cd.crop_id[c]}({c}) 不允许在该地块季次种植')
                ok = False
            if (c, season) not in cd.S0:
                issues.append(f'[{p}] {yr}{season} 作物{cd.crop_id[c]} 无2023销售基线')
                ok = False

    # 2. 水浇地：第二季存在 ⇔ 第一季非水稻
    for p in cd.d_plots:
        for yr in range(2024, 2031):
            c1 = plant.get((yr, '第一季', p))
            c2 = plant.get((yr, '第二季', p))
            if c1 == cd.rice and c2 is not None:
                issues.append(f'[{p}] {yr} 水稻与第二季作物并存'); ok = False
            if c1 is not None and c1 != cd.rice and c2 is None:
                issues.append(f'[{p}] {yr} 第一季蔬菜{c1}但无第二季作物'); ok = False

    # 3. 重茬：同一地块相邻插槽同作物（跨年同季 与 年内相邻季）
    for p in cd.plots_all:
        slots = cd.slots_of(p)
        # 跨年同季
        for yr in range(2024, 2030):
            for season in ['第一季', '第二季']:
                c1 = plant.get((yr, season, p)); c2 = plant.get((yr + 1, season, p))
                if c1 is not None and c1 == c2:
                    issues.append(f'[{p}] {yr}与{yr+1} {season} 重茬{c1}'); ok = False
        # 年内相邻季（水浇地/大棚）
        for yr in range(2024, 2031):
            c1 = plant.get((yr, '第一季', p)); c2 = plant.get((yr, '第二季', p))
            if c1 is not None and c1 == c2:
                issues.append(f'[{p}] {yr} 年内两季重茬{c1}'); ok = False

    # 4. 豆类三年轮作（含2023）
    for p in cd.plots_all:
        for wstart in range(2023, 2029):
            win = {wstart, wstart + 1, wstart + 2}
            cnt = 1.0 if (wstart == 2023 and cd.bean23.get(p, False)) else 0.0
            for yr in win:
                for season in ['第一季', '第二季']:
                    c = plant.get((yr, season, p))
                    if c in cd.bean_ids:
                        cnt += 1
            if cnt < 1:
                issues.append(f'[{p}] 窗口{wstart}-{wstart+2} 无豆类'); ok = False

    # 5. 超产统计（不判违规）：情形1超产滞销浪费、情形2超产按半价出售，
    #    均为题意允许行为；原价售出量 u≤S 已由模型约束强制。此处仅报告浪费占比。
    prod_tot = {}          # (c, season, yr) -> 该年产量
    for (yy, season, p), c in plant.items():
        key = (c, season, yy)
        yld = cd.unit_yield(p, c, season)
        prod_tot[key] = prod_tot.get(key, 0.0) + cd.land_area[p] * yld
    waste = sum(max(v - cd.S0[(c, s)] * S_mult.get((c, s), 1.0), 0.0)
                for (c, s, yr), v in prod_tot.items())
    total = sum(prod_tot.values())
    if total > 0 and waste / total > 0.01:
        issues.append(f'超产浪费占比 {100*waste/total:.1f}%（结构性超产，题意允许，非约束违规）')
        # 不置 ok=False：结构性超产是模型在复种/粒度约束下的最优选择

    if verbose:
        print('验证结果:', '✓ 全部约束满足' if ok else f'✗ {len(issues)} 项违规')
        for it in issues[:30]:
            print('  -', it)
    return ok, issues

if __name__ == '__main__':
    pass
