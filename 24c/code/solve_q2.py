# -*- coding: utf-8 -*-
"""问题2：两阶段随机规划 + CVaR 风险控制。
情景：销售量(小麦玉米+5-10%/年,其他±5%)、亩产量(±10%)、价格(粮食稳/蔬菜+5%/食用菌-1~-5%)、成本(+5%/年)
求解不同风险偏好 λ 的方案，评估风险指标，输出 result2.xlsx。
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time
import numpy as np
sys.path.insert(0, 'code')
import crops_data as cd
import scenarios as sc
from milp_model import PlantingMILP

YEARS = list(range(2024, 2031))
N = int(os.environ.get('N_SCEN', '20'))
OUT_TAG = os.environ.get('TAG', 'q2')

def solve_stoch(sales_mult, yield_mult, price, lambda_risk, tag, time_limit=1500, gap=0.03):
    cost = sc.cost_val_q23()
    m = PlantingMILP(mode='stoch', years=YEARS, sales_mult=sales_mult,
                     yield_mult=yield_mult, price_val=price, cost_val=cost,
                     lambda_risk=lambda_risk, alpha=0.95,
                     time_limit=time_limit, mip_rel_gap=gap)
    print(f'[{tag}] λ={lambda_risk} 变量={m.n_col} 约束={m.n_rows} 二元={len(m.ycol)} 情景={m.n_scen}')
    t0 = time.time()
    res = m.solve()
    print(f'  [{tag}] λ={lambda_risk} 求解 {time.time()-t0:.0f}s, 目标={m.objval:,.0f}, '
          f'gap={getattr(res, "mip_gap", None)}')
    if m.x is None:
        return None
    # 风险指标
    prf = {s: m.x[m.prf[s]] for s in range(m.n_scen)}
    prof = np.array([prf[s] for s in range(m.n_scen)])
    E = prof.mean(); sd = prof.std(); worst = prof.min()
    alpha = 0.95
    loss = -prof
    eta = m.x[m.eta_col]
    cvar = eta + (1/(1-alpha)) * np.mean(np.maximum(loss - eta, 0))
    loss_prob = (prof < 0).mean()
    print(f'  [{tag}] E[收益]={E:,.0f} 元, 标准差={sd:,.0f}, 最差={worst:,.0f}, '
          f'CVaR(损失)={cvar:,.0f}, 亏损概率={loss_prob:.3f}')
    plant, area = m.plan()
    json.dump({'tag': tag, 'lambda': lambda_risk, 'N': m.n_scen,
               'E_profit': float(E), 'std': float(sd), 'worst': float(worst),
               'cvar': float(cvar), 'loss_prob': float(loss_prob),
               'objval': m.objval,
               'per_scenario_profit': prof.tolist(),
               'plant': {f'{yr}|{s}|{p}': c for (yr, s, p), c in plant.items()}},
              open(f'out/plan_{tag}_l{lambda_risk}.json', 'w'), ensure_ascii=False)
    return m, plant

if __name__ == '__main__':
    os.makedirs('out', exist_ok=True)
    seed = int(os.environ.get('SEED', '2024'))
    sales_mult, yield_mult, price, _cm = sc.gen_q2(N, seed=seed)
    np.save(f'out/{OUT_TAG}_scenarios_sales.npy',
            np.array([sales_mult[(s, k, yr)] for s in range(N)
                      for k in sorted(cd.crop_season_pairs) for yr in YEARS]))
    lambdas = [float(x) for x in os.environ.get('LAMBDAS', '0.0,0.3,0.8').split(',')]
    results = {}
    for lam in lambdas:
        r = solve_stoch(sales_mult, yield_mult, price, lam, OUT_TAG,
                        time_limit=int(os.environ.get('TL', '1500')))
        if r is not None:
            results[lam] = r
    # 写入 result2.xlsx（推荐取 λ=0.3 的折中方案）
    from write_results import write_result
    pref = 0.3 if 0.3 in results else next(iter(results), None)
    if pref is not None:
        m, plant = results[pref]
        write_result(plant, 'result2.xlsx')
        print(f'[result2] 已用 λ={pref} 方案写入 result2.xlsx')
