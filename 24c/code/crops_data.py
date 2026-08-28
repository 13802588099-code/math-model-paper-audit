# -*- coding: utf-8 -*-
"""2024国赛C题 数据模块：统一加载附件，构建优化模型所需的所有参数。
对外导出：地块、作物、产量/成本/价格表、2023种植、预期销售量(2023产量)、豆类2023、约束结构。
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

C = '/Users/lzt778899/2026数模国赛/2024赛题/C题'
ATT1 = f'{C}/附件1.xlsx'; ATT2 = f'{C}/附件2.xlsx'

# ---------------- 1. 地块 ----------------
def load_land():
    land = pd.ExcelFile(ATT1).parse('乡村的现有耕地')
    land = land.dropna(subset=['地块名称'])
    land['地块类型'] = land['地块类型'].str.strip()
    land['面积'] = land['地块面积/亩'].astype(float)
    return land

land = load_land()
land_area = dict(zip(land['地块名称'], land['面积']))
land_type = dict(zip(land['地块名称'], land['地块类型']))
plots_all = list(land['地块名称'])                     # 54 地块
type_plots = {t: [p for p in plots_all if land_type[p] == t] for t in
              ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']}
grain_plots = type_plots['平旱地'] + type_plots['梯田'] + type_plots['山坡地']
d_plots = type_plots['水浇地']; e_plots = type_plots['普通大棚']; f_plots = type_plots['智慧大棚']

# ---------------- 2. 作物 ----------------
def load_crops():
    cr = pd.ExcelFile(ATT1).parse('乡村种植的农作物')
    cr = cr[pd.to_numeric(cr['作物编号'], errors='coerce').notna()]
    cr['作物编号'] = cr['作物编号'].astype(int)
    return cr

crops = load_crops()
crop_id = dict(zip(crops['作物编号'], crops['作物名称'].str.strip()))
crop_type = dict(zip(crops['作物编号'], crops['作物类型'].str.strip()))
crop_ids_all = sorted(crop_id)

# 作物分组
grains   = list(range(1, 16))       # 粮食 1-15（含豆类）
rice     = 16
veg_first= list(range(17, 35))      # 蔬菜 17-34（可种第一季）
veg_water_second = [35, 36, 37]     # 大白菜/白萝卜/红萝卜（仅水浇地第二季）
mushroom = [38, 39, 40, 41]         # 食用菌（仅普通大棚第二季）
bean_ids = [1, 2, 3, 4, 5, 17, 18, 19]   # 豆类作物
veg_ids  = list(range(17, 38))      # 蔬菜 17-37

# 智慧大棚第二季与第一季相同（18种蔬菜，无大白菜/白萝卜/红萝卜）
# 普通大棚第一季蔬菜 = veg_first；智慧大棚第一季蔬菜 = veg_first

# ---------------- 3. 统计表（产量/成本/价格） ----------------
def load_stat():
    st = pd.ExcelFile(ATT2).parse('2023年统计的相关数据')
    st = st.dropna(subset=['作物编号', '地块类型'])
    st['地块类型'] = st['地块类型'].str.strip()
    st['季次'] = st['种植季次'].str.strip().replace({'单季': '第一季'})
    st['亩产量'] = st['亩产量/斤'].astype(float)
    st['成本'] = st['种植成本/(元/亩)'].astype(float)
    st['价格'] = (st['销售单价/(元/斤)'].str.split('-').str[0].astype(float)
                  + st['销售单价/(元/斤)'].str.split('-').str[1].astype(float)) / 2
    st['价格低'] = st['销售单价/(元/斤)'].str.split('-').str[0].astype(float)
    st['价格高'] = st['销售单价/(元/斤)'].str.split('-').str[1].astype(float)
    st['key'] = list(zip(st['作物编号'].astype(int), st['地块类型'], st['季次']))
    return st

stat = load_stat()
yield_map = dict(zip(stat['key'], stat['亩产量']))
cost_map  = dict(zip(stat['key'], stat['成本']))
# 价格按 (作物, 季次)：同季内不同地块类型价格一致（已核验），直接覆盖即可
price_season, price_lo_season, price_hi_season = {}, {}, {}
for _, r in stat.iterrows():
    cid, season = int(r['作物编号']), r['季次']
    price_season[(cid, season)] = r['价格']
    price_lo_season[(cid, season)] = r['价格低']
    price_hi_season[(cid, season)] = r['价格高']
# 每作物默认价 = 第一季价（粮食/蔬菜主销季；食用菌/水浇地二季菜只有单季）
price_crop = {}
price_lo_crop, price_hi_crop = {}, {}
for cid in crop_ids_all:
    for season in ['第一季', '第二季']:
        if (cid, season) in price_season:
            price_crop[cid] = price_season[(cid, season)]
            price_lo_crop[cid] = price_lo_season[(cid, season)]
            price_hi_crop[cid] = price_hi_season[(cid, season)]
            break
# 智慧大棚第一季 = 普通大棚第一季（附件说明）
for cid in veg_first:
    k1 = (cid, '智慧大棚', '第一季'); k2 = (cid, '普通大棚', '第一季')
    if k2 in yield_map and k1 not in yield_map:
        yield_map[k1] = yield_map[k2]; cost_map[k1] = cost_map[k2]

# ---------------- 4. 2023种植 -> 预期销售量 & 2023豆类 ----------------
def load_plant23():
    pl = pd.ExcelFile(ATT2).parse('2023年的农作物种植情况')
    pl['种植地块'] = pl['种植地块'].ffill()
    pl['地块'] = pl['种植地块'].str.strip()
    pl['作物编号'] = pl['作物编号'].astype(int)
    pl['面积'] = pl['种植面积/亩'].astype(float)
    pl['季次'] = pl['种植季次'].str.strip().replace({'单季': '第一季'})
    return pl

plant23 = load_plant23()
bean23 = {p: False for p in plots_all}
for _, r in plant23.iterrows():
    if int(r['作物编号']) in bean_ids:
        bean23[r['地块']] = True

# 每 (作物, 季次) 2023产量 = 预期销售量基准 S
_S = {}
for _, r in plant23.iterrows():
    cid, ltype, season, area = int(r['作物编号']), land_type[r['地块']], r['季次'], r['面积']
    y = yield_map.get((cid, ltype, season), np.nan)
    if pd.isna(y):
        print(f"[warn] 无产量数据: 作物{cid} 地块{r['地块']}({ltype}) 季次{season}")
        continue
    _S[(cid, season)] = _S.get((cid, season), 0.0) + area * y
S0 = {(c, s): v for (c, s), v in _S.items()}
crop_season_pairs = sorted(S0.keys())   # 全部 (作物, 季次) 销售单位

# ---------------- 5. 各地块各季可种作物 ----------------
def allowed_crops(p, season):
    t = land_type[p]
    if p in grain_plots and season == '第一季':
        return grains
    if t == '水浇地':
        if season == '第一季':
            return [rice] + veg_first          # 水稻 或 第一季蔬菜
        else:
            return veg_water_second            # 大白菜/白萝卜/红萝卜
    if t == '普通大棚':
        return veg_first if season == '第一季' else mushroom
    if t == '智慧大棚':
        return veg_first                       # 两季均可种 18 种蔬菜
    return []

# 地块-季次-年 的插槽序列
SEASONS = ['第一季', '第二季']

def slots_of(p):
    """返回该地块 2024-2030 年的 (年份, 季次) 插槽列表（单季地块只有第一季）。"""
    t = land_type[p]
    if p in grain_plots:
        return [(yr, '第一季') for yr in range(2024, 2031)]
    return [(yr, s) for yr in range(2024, 2031) for s in SEASONS]

def grain_pm(p, c):
    """单季粮食地块每亩利润(元)。"""
    t = land_type[p]
    return price_season[(c, '第一季')] * yield_map[(c, t, '第一季')] - cost_map[(c, t, '第一季')]

def pm(p, c, season):
    """地块 p 季次 season 种作物 c 的每亩毛利润（原价销售口径）。"""
    t = land_type[p]
    return price_season[(c, season)] * yield_map[(c, t, season)] - cost_map[(c, t, season)]

def unit_cost(p, c, season):
    """每亩成本。"""
    return cost_map[(c, land_type[p], season)]

def unit_yield(p, c, season):
    return yield_map[(c, land_type[p], season)]

def unit_price(c, season='第一季'):
    """作物 c 在 season 的销售价(元/斤)。"""
    if (c, season) in price_season:
        return price_season[(c, season)]
    return price_crop[c]

# ---------------- 年度价格/成本演化（问题2/3） ----------------
def price_evolve(c, season, yr, mu_eps=0.0):
    """作物 c 在 yr 年、季次 season 的价格（相对2023）。粮食稳定；蔬菜+5%/年；食用菌-3%/年(羊肚菌-5%)。"""
    base = price_season[(c, season)]
    k = yr - 2023
    if c in grains + [rice]:
        return base * (1.0 + mu_eps)          # 粮食价格基本稳定
    if c in veg_ids:
        return base * (1.05 ** k) * (1.0 + mu_eps)
    if c == 41:
        return base * (0.95 ** k)             # 羊肚菌 -5%/年
    return base * (0.97 ** k)                 # 其他食用菌 -3%/年

def cost_evolve(p, c, season, yr):
    """成本年增5%。"""
    return unit_cost(p, c, season) * (1.05 ** (yr - 2023))

def sales_evolve(c, season, yr, eps=0.0, growth=0.0):
    """预期销售量演化：小麦/玉米年增 growth(5-10%)，其他作物±eps(±5%)。"""
    base = S0[(c, season)]
    k = yr - 2023
    if c in (6, 7):                            # 小麦、玉米
        return base * (1.0 + growth) ** k
    return base * (1.0 + eps)                  # 相对2023 ±5%

if __name__ == '__main__':
    print('地块数:', len(plots_all))
    print('作物数:', len(crop_ids_all))
    print('销售单位(作物,季次):', len(crop_season_pairs))
    print('豆类地块(2023):', sum(bean23.values()), '/', len(plots_all))
    print('小麦销售基准(斤):', S0[(6, '第一季')])
