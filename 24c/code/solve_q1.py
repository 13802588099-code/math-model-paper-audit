# -*- coding: utf-8 -*-
"""问题1：确定性优化，两种情形：
(1) 超产滞销浪费  -> result1_1.xlsx
(2) 超产按50%价出售 -> result1_2.xlsx
同时保存方案 JSON 供绘图与论文使用。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json, sys, time
sys.path.insert(0, 'code')
import crops_data as cd
from milp_model import PlantingMILP

YEARS = list(range(2024, 2031))

def build_params(mode):
    """构建确定性价格/成本参数（2023稳定）。"""
    price = {}
    cost = {}
    for (c, season) in cd.crop_season_pairs:
        for yr in YEARS:
            price[(c, season, yr)] = cd.unit_price(c, season)
    for p in cd.plots_all:
        for (yr, season) in cd.slots_of(p):
            for c in cd.allowed_crops(p, season):
                if (c, season) not in cd.S0:
                    continue
                cost[(p, c, season, yr)] = cd.unit_cost(p, c, season)
    return price, cost

def solve_case(mode, tag):
    price, cost = build_params(mode)
    m = PlantingMILP(mode=mode, years=YEARS, price_val=price, cost_val=cost,
                     time_limit=900, mip_rel_gap=0.01)
    t0 = time.time()
    res = m.solve()
    dt = time.time() - t0
    gap = m.gap if m.gap is not None else (res.mip_gap if res.success else 'N/A')
    print(f"[{tag}] 求解时间 {dt:.1f}s, 目标(总收益) = {m.objval:,.0f} 元, "
          f"gap={gap}")
    if m.x is None:
        print('  失败原因:', res.message)
        return None
    plant, area = m.plan()
    prod = m.production()
    # 保存方案
    json.dump({'mode': mode, 'tag': tag,
               'profit': m.objval,
               'plant': {f'{yr}|{s}|{p}': c for (yr, s, p), c in plant.items()},
               'production': {f'{c}|{s}|{yr}': v for (c, s, yr), v in prod.items()}},
              open(f'out/plan_{tag}.json', 'w'), ensure_ascii=False)
    # 写入结果模板
    from write_results import write_result
    out = f'result1_{tag[-1]}.xlsx' if tag.startswith('q1_') else f'result_{tag}.xlsx'
    write_result(plant, out)
    # 约束校验
    from validate_plan import validate
    ok, issues = validate(plant, verbose=False, mode=mode)
    print(f'  [校验] {tag}: {"全部约束满足" if ok else f"{len(issues)}项违规"}')
    if not ok:
        for it in issues[:10]:
            print('   -', it)
    return m, plant, area, prod

if __name__ == '__main__':
    import os
    os.makedirs('out', exist_ok=True)
    cases = {'q1_1': ('waste', 'q1_1'), 'q1_2': ('discount', 'q1_2')}
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which == 'both':
        m1, plant1, area1, prod1 = solve_case('waste', 'q1_1')
        m2, plant2, area2, prod2 = solve_case('discount', 'q1_2')
    else:
        solve_case(*cases[which])
