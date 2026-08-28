# -*- coding: utf-8 -*-
"""核心 MILP 建模求解模块（HiGHS via scipy.optimize.milp）。

统一框架：
- 决策变量 y[(p, yr, season, c)] ∈ {0,1}：地块 p 在 (年, 季) 是否种作物 c（单作物占满整个地块）
- 销售拆分变量 u（原价售出斤数）、v（打折/超出斤数）
- 情景 s 仅影响销售上限与产量的随机系数；y 为 here-and-now 共同决策

模式：
  'waste'   : 超产滞销，收益 = price·u, u ≤ min(prod, S)
  'discount': 超产 50% 价，收益 = price·u + 0.5·price·v, u+v=prod, u ≤ S
  'stoch'   : 多情景，目标 = 期望收益 - λ·CVaR_α(损失)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import crops_data as cd


class PlantingMILP:
    def __init__(self, mode, years, sales_mult=None, yield_mult=None,
                 price_val=None, cost_val=None, S0=None,
                 lambda_risk=0.0, alpha=0.95, weights=None, time_limit=300,
                 mip_rel_gap=1e-4):
        self.mode = mode
        self.years = list(years)
        self.S0 = S0 if S0 is not None else cd.S0
        self.n_scen = 1 if mode in ('waste', 'discount') else \
            max(k[0] for k in sales_mult) + 1
        self.sales_mult = sales_mult or {}
        self.yield_mult = yield_mult or {}
        self.price = price_val or {}
        self.cost = cost_val or {}
        self.lambda_risk = lambda_risk
        self.alpha = alpha
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap
        if weights is not None:
            self.w = {int(k): v for k, v in weights.items()}
        else:
            self.w = {s: 1.0 / self.n_scen for s in range(self.n_scen)}
        self._build()

    # ================= 建模 =================
    def _build(self):
        col = 0
        self.ycol = {}                 # (p, yr, season, c) -> col
        self.u = {}                    # (s, k, yr) -> col
        self.v = {}                    # (s, k, yr) -> col (discount)
        self.klist = sorted(cd.crop_season_pairs)

        self.slots = {p: cd.slots_of(p) for p in cd.plots_all}

        # 1) 二进制种植变量（仅保留有2023销售基线的组合）
        #    剪枝：waste 模式下，单地块产量已超总销售上限的组合不可行，剔除
        for p in cd.plots_all:
            for (yr, season) in self.slots[p]:
                for c in cd.allowed_crops(p, season):
                    if (c, season) not in self.S0:
                        continue
                    self.ycol[(p, yr, season, c)] = col
                    col += 1

        # 2) 销售变量
        for s in range(self.n_scen):
            for (c, season) in self.klist:
                for yr in self.years:
                    self.u[(s, (c, season), yr)] = col; col += 1
                    if self.mode == 'discount':
                        self.v[(s, (c, season), yr)] = col; col += 1
        self.n_col = col

        self.integrality = np.zeros(self.n_col)
        for idx in self.ycol.values():
            self.integrality[idx] = 1

        self._row_inds, self._row_vals, self._row_lb, self._row_ub = [], [], [], []

        def R(lb_, ub_, d):
            if not d:
                return
            cols = np.array(sorted(d.keys()), dtype=int)
            vals = np.array([d[c] for c in cols], dtype=float)
            self._row_inds.append(cols)
            self._row_vals.append(vals)
            self._row_lb.append(lb_)
            self._row_ub.append(ub_)

        # ---------- 约束1：每地块每季至多一种作物 ----------
        for p in cd.plots_all:
            for (yr, season) in self.slots[p]:
                d = {}
                for c in cd.allowed_crops(p, season):
                    key = (p, yr, season, c)
                    if key in self.ycol:
                        d[self.ycol[key]] = 1.0
                R(0, 1, d)

        # ---------- 约束2：水浇地 第二季作物数 = 1 - 水稻 ----------
        for p in cd.d_plots:
            for yr in self.years:
                d = {}
                for c in cd.veg_water_second:
                    key = (p, yr, '第二季', c)
                    if key in self.ycol:
                        d[self.ycol[key]] = 1.0
                krice = (p, yr, '第一季', cd.rice)
                if krice in self.ycol:
                    d[self.ycol[krice]] = 1.0
                R(1, 1, d)

        # ---------- 约束3a：重茬-跨年同季（同作物连续两年同地块） ----------
        for p in cd.plots_all:
            for (yr, season) in self.slots[p]:
                if yr == max(self.years):
                    continue
                for c in cd.allowed_crops(p, season):
                    k1 = (p, yr, season, c); k2 = (p, yr + 1, season, c)
                    if k1 in self.ycol and k2 in self.ycol:
                        R(0, 1, {self.ycol[k1]: 1.0, self.ycol[k2]: 1.0})

        # ---------- 约束3b：重茬-年内相邻季（智慧大棚两季同作物互斥） ----------
        for p in cd.f_plots:
            for yr in self.years:
                for c in cd.allowed_crops(p, '第一季'):
                    k1 = (p, yr, '第一季', c); k2 = (p, yr, '第二季', c)
                    if k1 in self.ycol and k2 in self.ycol:
                        R(0, 1, {self.ycol[k1]: 1.0, self.ycol[k2]: 1.0})

        # ---------- 约束4：豆类三年轮作（每个滚动三年窗口至少一次豆类，2023计入首窗） ----------
        for p in cd.plots_all:
            for wstart in range(2023, 2029):        # 2023-25 ... 2028-30
                win = {wstart, wstart + 1, wstart + 2}
                d = {}
                for (yr, season) in self.slots[p]:
                    if yr not in win:
                        continue
                    for c in cd.bean_ids:
                        key = (p, yr, season, c)
                        if key in self.ycol:
                            d[self.ycol[key]] = 1.0
                # 需求：每个窗口至少1次豆类；仅当2023年已种豆类时首窗(2023-25)可豁免
                need = 0.0 if (wstart == 2023 and cd.bean23.get(p, False)) else 1.0
                R(need, np.inf, d)

        # ---------- 约束5：产量-销售关系 ----------
        for s in range(self.n_scen):
            for (c, season) in self.klist:
                for yr in self.years:
                    k = (c, season)
                    ymult = self.yield_mult.get((s, c, yr), 1.0)
                    d = {}
                    for p in cd.plots_all:
                        key = (p, yr, season, c)
                        if key in self.ycol:
                            d[self.ycol[key]] = cd.land_area[p] * \
                                cd.unit_yield(p, c, season) * ymult
                    S = self.S0[k] * self.sales_mult.get((s, k, yr), 1.0)
                    ucol = self.u[(s, k, yr)]
                    if self.mode == 'discount':
                        vcol = self.v[(s, k, yr)]
                        d[ucol] = -1.0
                        d[vcol] = -1.0
                        R(0, 0, d)                  # u + v = prod
                        R(0, S, {ucol: 1.0})        # u ≤ S
                    else:
                        d[ucol] = -1.0
                        R(0, np.inf, d)             # prod - u ≥ 0  => u ≤ prod
                        R(0, S, {ucol: 1.0})        # u ≤ S

        # ================= 目标 =================
        if self.mode in ('waste', 'discount'):
            s = 0
            self.obj_ = np.zeros(self.n_col)
            for (c, season) in self.klist:
                for yr in self.years:
                    pr = self.price.get((c, season, yr), cd.unit_price(c, season))
                    self.obj_[self.u[(s, (c, season), yr)]] += pr
                    if self.mode == 'discount':
                        self.obj_[self.v[(s, (c, season), yr)]] += 0.5 * pr
            for (p, yr, season, c), idx in self.ycol.items():
                self.obj_[idx] -= self.cost[(p, c, season, yr)] * cd.land_area[p]
            self.obj = -self.obj_
        else:
            # 随机规划：max E[prf] - λ·CVaR
            self.prf = {}
            for s in range(self.n_scen):
                self.prf[s] = self.n_col; self.n_col += 1
            self.eta_col = self.n_col; self.n_col += 1
            self.zeta = {}
            for s in range(self.n_scen):
                self.zeta[s] = self.n_col; self.n_col += 1
            self.integrality = np.concatenate([self.integrality,
                                               np.zeros(self.n_col - len(self.integrality))])
            self.obj = np.zeros(self.n_col)

            # prf[s] = Σ price·u - Σ cost·A·y
            for s in range(self.n_scen):
                d = {self.prf[s]: -1.0}
                for (c, season) in self.klist:
                    for yr in self.years:
                        pr = self.price.get((s, c, season, yr),
                                            self.price.get((c, season, yr),
                                                           cd.unit_price(c, season)))
                        d[self.u[(s, (c, season), yr)]] = pr
                for (p, yr, season, c), idx in self.ycol.items():
                    cst = self.cost.get((s, p, c, season, yr),
                                        self.cost.get((p, c, season, yr), 0.0))
                    d[idx] = -cst * cd.land_area[p]
                R(0, 0, d)
            # ζ_s ≥ -prf_s - η
            for s in range(self.n_scen):
                R(0, np.inf, {self.zeta[s]: 1.0, self.prf[s]: 1.0, self.eta_col: 1.0})
            # 目标(最小化) = -Σ w·prf + λ·η + λ/(1-α)·Σ w·ζ
            for s in range(self.n_scen):
                self.obj[self.prf[s]] -= self.w[s]
            self.obj[self.eta_col] += self.lambda_risk
            for s in range(self.n_scen):
                self.obj[self.zeta[s]] += self.lambda_risk / (1 - self.alpha) * self.w[s]

        self._assemble()

    def _assemble(self):
        self.n_rows = len(self._row_lb)
        self.A = np.zeros((self.n_rows, self.n_col))
        for i, cols in enumerate(self._row_inds):
            self.A[i, cols] = self._row_vals[i]
        self.lb = np.array(self._row_lb, dtype=float)
        self.ub = np.array(self._row_ub, dtype=float)
        # 剔除完全无约束意义的行 (lb=-inf, ub=+inf)
        keep = np.isfinite(self.lb) | np.isfinite(self.ub)
        if not keep.all():
            self.A = self.A[keep]; self.lb = self.lb[keep]; self.ub = self.ub[keep]

    # ================= 求解 =================
    def solve(self, msg=True):
        ub = np.full(self.n_col, 1e9)
        ub[np.where(self.integrality == 1)[0]] = 1.0
        lb = np.zeros(self.n_col)
        # CVaR 线性化的 η 变量必须自由（VaR 可为负），否则 CVaR 被强制 ≥0
        if self.mode == 'stoch':
            lb[self.eta_col] = -np.inf
        bounds = Bounds(lb, ub)
        res = milp(c=self.obj,
                   constraints=LinearConstraint(self.A, self.lb, self.ub),
                   integrality=self.integrality, bounds=bounds,
                   options={'time_limit': self.time_limit,
                            'mip_rel_gap': self.mip_rel_gap,
                            'presolve': True})
        self.res = res
        # HiGHS 超时或达到 gap 也可能返回可行解
        if res.x is not None:
            self.x = res.x
            self.objval = -res.fun
            self.gap = getattr(res, 'mip_gap', None)
        else:
            self.x = None
            self.objval = None
            self.gap = None
            if msg:
                print('[MILP] 无可行解:', res.message)
        if msg and res.x is not None and not res.success:
            print(f'[MILP] 超时/中止, 采用当前可行解, 目标={self.objval:,.0f}, '
                  f'gap={self.gap}')
        return res

    # ================= 结果提取 =================
    def plan(self):
        """种植方案: dict[(yr, season, p)] = c 及 面积 dict"""
        plant, area = {}, {}
        if self.x is None:
            return plant, area
        for (p, yr, season, c), idx in self.ycol.items():
            if self.x[idx] > 0.5:
                plant[(yr, season, p)] = c
                area[(p, yr, season, c)] = cd.land_area[p]
        return plant, area

    def production(self, scenario=0):
        """各(作物,季次,年)产量(斤)，确定性口径（yield_mult=1）"""
        out = {}
        if self.x is None:
            return out
        for (c, season) in self.klist:
            for yr in self.years:
                prod = 0.0
                for p in cd.plots_all:
                    key = (p, yr, season, c)
                    if key in self.ycol and self.x[self.ycol[key]] > 0.5:
                        prod += cd.land_area[p] * cd.unit_yield(p, c, season)
                out[(c, season, yr)] = prod
        return out

    def profit(self):
        """确定性总收益(元)"""
        if self.x is None:
            return 0.0
        tot = 0.0
        for (c, season) in self.klist:
            for yr in self.years:
                pr = self.price.get((c, season, yr), cd.unit_price(c, season))
                tot += pr * self.x[self.u[(0, (c, season), yr)]]
                if self.mode == 'discount':
                    tot += 0.5 * pr * self.x[self.v[(0, (c, season), yr)]]
        for (p, yr, season, c), idx in self.ycol.items():
            tot -= self.cost[(p, c, season, yr)] * cd.land_area[p] * self.x[idx]
        return tot

    def sold(self, k, yr):
        """原价销量(斤) 确定情形"""
        return self.x[self.u[(0, k, yr)]]

    def excess(self, k, yr):
        """超出量(斤) 确定情形（discount 为 v，其余为 prod-u）"""
        if self.mode == 'discount':
            return self.x[self.v[(0, k, yr)]]
        prod = self.production().get((k[0], k[1], yr), 0.0)
        return max(prod - self.sold(k, yr), 0.0)
