# -*- coding: utf-8 -*-
"""问题3：相关情景模拟优化。
在问题2基础上引入作物间替代/互补、销售量-价格、产量-价格等相关系结构
（Cholesky 分解生成相关正态情景），求解并评估，与问题2独立情景方案对比。
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
OUT_TAG = os.environ.get('TAG', 'q3')

def solve_stoch(sales_mult, yield_mult, price, lambda_risk, tag, time_limit=1500, gap=0.04):
    cost = sc.cost_val_q23()
    m = PlantingMILP(mode='stoch', years=YEARS, sales_mult=sales_mult,
                     yield_mult=yield_mult, price_val=price, cost_val=cost,
                     lambda_risk=lambda_risk, alpha=0.05,
                     time_limit=time_limit, mip_rel_gap=gap)
    print(f'[{tag}] λ={lambda_risk} 变量={m.n_col} 约束={m.n_rows} 二元={len(m.ycol)} 情景={m.n_scen}')
    t0 = time.time()
    res = m.solve()
    print(f'  [{tag}] λ={lambda_risk} 求解 {time.time()-t0:.0f}s, 目标={m.objval:,.0f}')
    if m.x is None:
        return None
    prf = {s: m.x[m.prf[s]] for s in range(m.n_scen)}
    prof = np.array([prf[s] for s in range(m.n_scen)])
    E = prof.mean(); sd = prof.std(); worst = prof.min()
    alpha = 0.05
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
    seed = int(os.environ.get('SEED', '7'))
    sales_mult, yield_mult, price, _cm = sc.gen_q3(N, seed=seed)
    lambdas = [float(x) for x in os.environ.get('LAMBDAS', '0.3,0.8').split(',')]
    for lam in lambdas:
        solve_stoch(sales_mult, yield_mult, price, lam, OUT_TAG,
                    time_limit=int(os.environ.get('TL', '1500')))
