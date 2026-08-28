# -*- coding: utf-8 -*-
"""问题2/3 随机规划求解（HiGHS 直连 + MIP 热启动）。

相比 scipy.optimize.milp 的改进：
- 用 highspy 直接调用 HiGHS，可传入 MIP 起点（Q1 情形1 的已知可行方案），
  使分支定界从好内点出发，避免长时间停在差方案；
- 求解中途在日志输出当前下界/内点/gap。

用法：
  N_SCEN=15 SEED=2024 TAG=q2 LAMBDAS=0.0,0.3,0.8 TL=600 GAP=0.03 \
    .venv/bin/python3 code/solve_stoch_hs.py
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, 'code')
import crops_data as cd
import scenarios as sc
from milp_model import PlantingMILP

YEARS = list(range(2024, 2031))
N = int(os.environ.get('N_SCEN', '15'))
OUT_TAG = os.environ.get('TAG', 'q2')
TL = int(os.environ.get('TL', '600'))
GAP = float(os.environ.get('GAP', '0.03'))
WARM = os.environ.get('WARM', 'q1_1')      # MIP 起点方案（None 则不用）
ALPHA = float(os.environ.get('ALPHA', '0.95'))   # CVaR 置信水平

def build_warm_start(m):
    """由已知可行方案构造 y 变量起点。"""
    if not WARM:
        return None
    p = f'out/plan_{WARM}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    start = np.zeros(m.n_col)
    for k, c in d['plant'].items():
        yr, s, p_ = k.split('|')
        key = (p_, int(yr), s, c)
        if key in m.ycol:
            start[m.ycol[key]] = 1.0
    return start

def highspy_solve(m, warm_start=None, time_limit=600, gap=0.03, seed=7):
    import highspy
    h = highspy.Highs()
    h.setOptionValue('time_limit', time_limit)
    h.setOptionValue('mip_rel_gap', gap)
    h.setOptionValue('parallel', 'on')
    h.setOptionValue('random_seed', seed)
    h.setOptionValue('mip_max_nodes', 2_000_000)
    h.setOptionValue('log_to_console', False)
    h.setOptionValue('output_flag', True)

    A = sp.csc_matrix(m.A)
    n, nc = m.n_rows, m.n_col
    lp = highspy.HighsLp()
    lp.num_col_ = nc
    lp.num_row_ = n
    lp.col_cost_ = m.obj
    lb = np.zeros(nc)
    if getattr(m, 'eta_col', None) is not None:
        lb[m.eta_col] = -1e9     # CVaR 的 η 变量自由（VaR 可为负）
    lp.col_lower_ = lb
    ub = np.full(nc, 1e9)
    ub[np.where(m.integrality == 1)[0]] = 1.0
    lp.col_upper_ = ub
    lp.row_lower_ = m.lb
    lp.row_upper_ = m.ub
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMinimize
    h.passModel(lp)
    # 整数性
    int_inds = np.where(m.integrality == 1)[0].astype(np.int32)
    if len(int_inds):
        h.changeColsIntegrality(len(int_inds), int_inds,
                                np.full(len(int_inds),
                                        highspy.HighsVarType.kInteger,
                                        dtype=np.uint8))
    # MIP 起点（仅提供 0-1 种植变量取值；连续变量留给求解器）
    if warm_start is not None:
        idx = np.where(warm_start > 0.5)[0].astype(np.int32)
        vals = np.ones(len(idx), dtype=np.float64)
        h.setSolution(len(idx), idx, vals)
    h.run()
    info = h.getInfo()
    x = h.getSolution().col_value
    x = np.array(x)
    status = info.mip_node_count
    return h, info, x, float(info.primal_solution_status if hasattr(info, 'primal_solution_status') else 0)

def evaluate(m, x):
    """由解向量计算风险指标。
    CVaR 用经验尾均值（最差 (1-α) 情景的平均损失，取 loss 最大的 n_tail 个），
    不依赖 η，避免 λ=0 时 η 退化。"""
    prf = {s: x[m.prf[s]] for s in range(m.n_scen)}
    prof = np.array([prf[s] for s in range(m.n_scen)])
    E = prof.mean(); sd = prof.std(); worst = prof.min()
    loss = -prof
    n_tail = max(1, int(np.ceil((1 - m.alpha) * m.n_scen)))
    cvar = float(np.sort(loss)[-n_tail:].mean())   # 最差 (1-α) 情景的平均损失
    loss_prob = (prof < 0).mean()
    return E, sd, worst, cvar, loss_prob, prof

def solve_stoch(sales_mult, yield_mult, price, cost_mult, lambda_risk, tag, time_limit, gap, seed):
    cost = sc.cost_val_q23(cost_mult, N)
    # alpha = 置信水平（RU 公式 1/(1-alpha) 为尾权重）；α 大则尾窄、目标易退化，
    # 取 α=0.90（尾 3/30 情景）兼顾风险刻画与求解稳定性
    m = PlantingMILP(mode='stoch', years=YEARS, sales_mult=sales_mult,
                     yield_mult=yield_mult, price_val=price, cost_val=cost,
                     lambda_risk=lambda_risk, alpha=ALPHA,
                     time_limit=time_limit, mip_rel_gap=gap)
    print(f'[{tag}] λ={lambda_risk} 变量={m.n_col} 约束={m.n_rows} '
          f'二元={len(m.ycol)} 情景={m.n_scen}', flush=True)
    warm = build_warm_start(m)
    if warm is not None:
        print(f'  [热启动] 来自 {WARM} 方案 ({int(warm.sum())} 个地块种植)', flush=True)
    t0 = time.time()
    h, info, x, _ = highspy_solve(m, warm, time_limit, gap, seed)
    dt = time.time() - t0
    if x is None or not np.isfinite(x).all():
        print(f'  [{tag}] λ={lambda_risk} 无可行解 ({dt:.0f}s)', flush=True)
        return None
    # 目标值（解的实际目标 = -Σ w prf + λ·CVaR 项，直接取 -c^T x 更稳）
    m.x = x
    m.objval = -m.obj @ x
    objval = m.objval
    E, sd, worst, cvar, loss_prob, prof = evaluate(m, x)
    bound = info.mip_gap if hasattr(info, 'mip_gap') else None
    print(f'  [{tag}] λ={lambda_risk} 求解 {dt:.0f}s, 目标={objval:,.0f}, '
          f'E={E:,.0f}, 最差={worst:,.0f}, CVaR={cvar:,.0f}, '
          f'亏损概率={loss_prob:.3f}', flush=True)
    plant, area = m.plan()
    json.dump({'tag': tag, 'lambda': lambda_risk, 'N': m.n_scen,
               'E_profit': float(E), 'std': float(sd), 'worst': float(worst),
               'cvar': float(cvar), 'loss_prob': float(loss_prob),
               'objval': float(objval), 'solve_time': dt,
               'per_scenario_profit': prof.tolist(),
               'plant': {f'{yr}|{s}|{p}': c for (yr, s, p), c in plant.items()}},
              open(f'out/plan_{tag}_l{lambda_risk}.json', 'w'), ensure_ascii=False)
    return m, plant

if __name__ == '__main__':
    os.makedirs('out', exist_ok=True)
    seed = int(os.environ.get('SEED', '2024'))
    if OUT_TAG == 'q2':
        sales_mult, yield_mult, price, cost_mult = sc.gen_q2(N, seed=seed)
    else:
        sales_mult, yield_mult, price, cost_mult = sc.gen_q3(N, seed=seed)
    lambdas = [float(x) for x in os.environ.get('LAMBDAS', '0.0,0.3,0.8').split(',')]
    results = {}
    for lam in lambdas:
        r = solve_stoch(sales_mult, yield_mult, price, cost_mult, lam, OUT_TAG, TL, GAP, seed)
        if r is not None:
            results[lam] = r
    # 写入 result2.xlsx（推荐 λ=0.3 折中方案）
    from write_results import write_result
    pref = 0.3 if 0.3 in results else next(iter(results), None)
    if pref is not None and OUT_TAG == 'q2':
        m, plant = results[pref]
        write_result(plant, 'result2.xlsx')
        print(f'[result2] 已用 λ={pref} 方案写入 result2.xlsx', flush=True)
