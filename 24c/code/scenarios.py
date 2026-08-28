# -*- coding: utf-8 -*-
"""问题2/3 的情景生成。

问题2（独立随机，作物间不相关）：
  每个情景 = 一条可能的 7 年实现路径。每作物一个持久水平冲击（跨年保持），
  叠加逐年独立噪声，幅度均在题目给定范围内：
    小麦/玉米 销售量年增 U(5%,10%)；其他作物销售量相对2023 水平±5% + 年噪声±5%
    亩产量 水平±10% + 年噪声±10%；成本 +5%/年 × 水平±5%
    价格：粮食稳(±2%)、蔬菜+5%/年(水平±3%+年噪声±3%)、食用菌-3%/年(羊肚菌-5%)
  独立冲击使 41 种作物的风险彼此无关，组合可充分分散 → 风险相对较小。

问题3（相关随机）：
  在问题2基础上引入作物间相关结构（Cholesky 分解生成相关正态情景）：
  - 共同因子：产量间(天气) +0.4、蔬菜价格间(市场) +0.3、
    非粮销售间(需求) +0.2、成本间 +0.2；
  - 替代组(销售 -0.6)、互补组(销售 +0.4)；
  - 销售量-价格 -0.5、产量-价格 -0.3、产量-销售 +0.2。
  相关结构使不利事件（低产/低价/滞销）在作物间同步发生 → 组合风险显著放大。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import crops_data as cd

YEARS = list(range(2024, 2031))


# ---------------- 问题2：独立情景 ----------------
def gen_q2(N, seed=42):
    """问题2：各作物冲击彼此独立（相关矩阵为单位阵）。
    与问题3共用同一套"边际冲击幅度"映射（_apply），唯一差别是不做 Cholesky 相关化，
    从而在保持各作物波动范围不变的前提下，仅分离出"作物间相关性"对组合风险的净效应。
    """
    rng = np.random.default_rng(seed)
    units = cd.crop_season_pairs
    D = len(units) + 3 * len(cd.crop_ids_all)     # 销售47 + 产量41 + 价格41 + 成本41
    Z = rng.standard_normal((D, N))
    Zc = np.clip(Z, -1.0, 1.0)
    return _apply(Zc, N)


# ---------------- 冲击向量 -> 情景参数字典 ----------------
def _apply(Zc, N):
    """将冲击矩阵 Zc (D×N) 映射为四个乘子/价格字典。
    变量顺序与 build_corr 一致：销售[47] 产量[41] 价格[41] 成本[41]。
    """
    units = cd.crop_season_pairs
    crops = list(cd.crop_ids_all)
    n_s, n_c = len(units), len(crops)
    sales_mult, yield_mult, price, cost_mult = {}, {}, {}, {}
    for s in range(N):
        z = Zc[:, s]
        for i, (c, season) in enumerate(units):
            for yr in YEARS:
                if c in (6, 7):
                    g = 0.075 + 0.025 * z[i]          # 增长与销售冲击耦合
                    sales_mult[(s, (c, season), yr)] = (1 + max(0.05, min(0.10, g))) ** (yr - 2023)
                else:
                    sales_mult[(s, (c, season), yr)] = 1 + 0.05 * z[i]
        for i, c in enumerate(crops):
            for yr in YEARS:
                yield_mult[(s, c, yr)] = 1 + 0.10 * z[n_s + i]
                cost_mult[(s, c, yr)] = 1 + 0.05 * z[n_s + n_c + n_c + i]
                sh = z[n_s + n_c + i]                  # 价格冲击每作物一个，两季共用
                for season in ['第一季', '第二季']:
                    if (c, season) not in cd.S0:
                        continue
                    base = cd.price_evolve(c, season, yr)
                    if c in cd.grains + [cd.rice]:
                        price[(s, c, season, yr)] = base * (1 + 0.02 * sh)
                    elif c in cd.veg_ids:
                        price[(s, c, season, yr)] = base * (1 + 0.03 * sh)
                    elif c == 41:
                        price[(s, c, season, yr)] = base
                    else:
                        price[(s, c, season, yr)] = base * (1 + 0.02 * sh)
    return sales_mult, yield_mult, price, cost_mult


# ---------------- 问题3：相关情景 ----------------
def build_corr():
    """构造随机向量的相关矩阵（共同因子 + 配对结构，半正定保证）。
    变量顺序：销售冲击 eps[cs]（47个），产量冲击 delta[c]（41个），
    价格冲击 pi[c]（41个），成本冲击 chi[c]（41个）

    机制：共同因子（天气/市场/成本）放大"同涨同跌"，替代对提供对冲，
    互补对与产销联动刻画供需联动。目标矩阵经 Higham 投影为合法相关矩阵。
    """
    units = cd.crop_season_pairs
    n_s = len(units)
    n_c = len(cd.crop_ids_all)
    D = n_s + n_c + n_c + n_c
    C = np.eye(D)
    crops = list(cd.crop_ids_all)

    def ci(c): return n_s + crops.index(c)
    def pi(c): return n_s + n_c + crops.index(c)
    def costi(c): return n_s + n_c + n_c + crops.index(c)
    def si(cs): return units.index(cs)
    def unit_of(c):                       # 该作物在 S0 中的(季次)单元
        for s in ('第一季', '第二季'):
            if (c, s) in cd.S0:
                return (c, s)
        return None
    def link_sales(c1, c2, r):           # 两作物销售单元之间设相关
        u1, u2 = unit_of(c1), unit_of(c2)
        if u1 and u2:
            C[si(u1), si(u2)] = r
            C[si(u2), si(u1)] = r

    # 1) 共同因子：产量间(天气) +0.35；蔬菜价格间(市场) +0.25；成本间 +0.15
    for i in range(n_c):
        for j in range(i + 1, n_c):
            C[ci(crops[i]), ci(crops[j])] = 0.35
            C[ci(crops[j]), ci(crops[i])] = 0.35
            C[costi(crops[i]), costi(crops[j])] = 0.15
            C[costi(crops[j]), costi(crops[i])] = 0.15
            if crops[i] in cd.veg_ids and crops[j] in cd.veg_ids:
                C[pi(crops[i]), pi(crops[j])] = 0.25
                C[pi(crops[j]), pi(crops[i])] = 0.25

    # 2) 替代组（销售量负相关）：两作物组全互斥；多成员组用"星形"（主导作物对成员）
    link_sales(6, 7, -0.6)                          # 小麦-玉米
    for lead, members in [(35, [36, 37]),           # 大白菜-白萝卜-红萝卜
                          (1, [2, 3, 4, 5]),        # 黄豆-黑豆红豆绿豆爬豆
                          (17, [18, 19]),           # 豇豆-刀豆-芸豆
                          (38, [39, 40, 41]),       # 香菇-金针菇平菇羊肚菌
                          (21, [24, 31])]:          # 西红柿-青椒-辣椒
        for m in members:
            link_sales(lead, m, -0.4)

    # 3) 互补组（销售量正相关 +0.4）：粮食↔蔬菜、蔬菜↔食用菌
    for c1, c2 in [(6, 21), (7, 20), (7, 22), (38, 21), (39, 25), (41, 17)]:
        link_sales(c1, c2, 0.4)

    # 4) 产销联动：销售量-价格 -0.5；产量-价格 -0.3；产量-销售量 +0.2
    for (c, season) in units:
        C[si((c, season)), pi(c)] = -0.5
        C[pi(c), si((c, season))] = -0.5
    for c in crops:
        C[ci(c), pi(c)] = -0.3
        C[pi(c), ci(c)] = -0.3
        for (c2, s2) in units:
            if c2 == c:
                C[ci(c), si((c, s2))] = 0.2
                C[si((c, s2)), ci(c)] = 0.2

    # 对称化 & 投影到"单位对角 + 半正定"的最近相关矩阵
    C = (C + C.T) / 2
    C = nearest_corr(C)
    return C, units, crops


def nearest_corr(A, max_iter=200, tol=1e-8):
    """Higham(2002) 交替投影：交替投影到半正定锥与单位对角集合，
    收敛到与原矩阵距离最近的合法相关矩阵（单位对角 + 半正定）。"""
    A = (A + A.T) / 2
    X, Y = A.copy(), A.copy()
    for _ in range(max_iter):
        R = Y
        eigval, eigvec = np.linalg.eigh(R)
        eigval = np.clip(eigval, 0.0, None)
        X = (eigvec * eigval) @ eigvec.T
        X = (X + X.T) / 2
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.norm(Y - X, 'fro') < tol:
            break
    # 末轮以半正定投影收尾并加微小脊保证 Cholesky 数值稳定
    eigval, eigvec = np.linalg.eigh(Y)
    eigval = np.clip(eigval, 0.0, None)
    C = (eigvec * eigval) @ eigvec.T
    C = (C + C.T) / 2
    return C + 1e-10 * np.eye(C.shape[0])


def gen_q3(N, seed=7):
    Cmat, units, crops = build_corr()
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Cmat)
    Z = L @ rng.standard_normal((Cmat.shape[0], N))   # D×N
    # 截断到 [-1,1]，保证冲击幅度在题目给定范围内（亩产量±10%、销售量±5%、价格±2~3%）
    Zc = np.clip(Z, -1.0, 1.0)
    return _apply(Zc, N)


def price_val_q23(price_shock=None):
    """确定性/分情景价格。返回 dict[(c,season,yr)] 或 dict[(s,c,season,yr)]"""
    out = {}
    for (c, season) in cd.crop_season_pairs:
        for yr in YEARS:
            out[(c, season, yr)] = cd.price_evolve(c, season, yr)
    if price_shock is not None:
        for (s, c), sh in price_shock.items():
            for season in ['第一季', '第二季']:
                if (c, season) in cd.S0:
                    for yr in YEARS:
                        out[(s, c, season, yr)] = cd.price_evolve(c, season, yr) * (1 + sh)
    return out


def cost_val_q23(cost_mult=None, N=1):
    """确定性成本 +5%/年；若给 cost_mult[(s,c,yr)] 则输出分情景成本 dict[(s,p,c,season,yr)]。"""
    cost = {}
    for p in cd.plots_all:
        for (yr, season) in cd.slots_of(p):
            for c in cd.allowed_crops(p, season):
                if (c, season) not in cd.S0:
                    continue
                base = cd.cost_evolve(p, c, season, yr)
                if cost_mult is not None:
                    for s in range(N):
                        cost[(s, p, c, season, yr)] = base * cost_mult.get((s, c, yr), 1.0)
                else:
                    cost[(p, c, season, yr)] = base
    return cost


if __name__ == '__main__':
    sm, ym, pr, cm = gen_q2(5)
    print('Q2 情景示例 小麦2024销售乘子:', sm[(0, (6, '第一季'), 2024)])
    sm3, ym3, pr3, cm3 = gen_q3(5)
    print('Q3 情景示例 小麦2024销售乘子:', sm3[(0, (6, '第一季'), 2024)])
    print('Q3 情景价格 羊肚菌2030:', pr3[(0, 41, '第二季', 2030)])
    # 各情景组合收益的标准差（用假定的单位收益 1 粗略看离散度）
    for name, s, y, p, c in [('Q2', sm, ym, pr, cm), ('Q3', sm3, ym3, pr3, cm3)]:
        prof = []
        for s_ in range(5):
            tot = sum(p.get((s_, cc, se, yr), 1.0) * s.get((s_, (cc, se), yr), 1.0)
                      for (cc, se) in cd.crop_season_pairs for yr in YEARS)
            tot += sum(y.get((s_, cc, yr), 1.0) for cc in cd.crop_ids_all for yr in YEARS)
            prof.append(tot)
        print(f'{name} 情景离散度(粗略): std={np.std(prof):.3f} 相对={np.std(prof)/np.mean(prof)*100:.1f}%')
