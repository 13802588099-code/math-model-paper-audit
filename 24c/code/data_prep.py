# -*- coding: utf-8 -*-
"""数据准备：读取附件，构建优化所需的作物/地块/产量/成本/价格/2023产量(预期销售量)表"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json

C = '/Users/lzt778899/2026数模国赛/2024赛题/C题'
ATT1 = f'{C}/附件1.xlsx'; ATT2 = f'{C}/附件2.xlsx'

# ---------- 附件1 地块 ----------
land = pd.ExcelFile(ATT1).parse('乡村的现有耕地')
land['地块类型'] = land['地块类型'].str.strip()
land = land.dropna(subset=['地块名称'])
land['面积'] = land['地块面积/亩'].astype(float)
land_type = dict(zip(land['地块名称'], land['地块类型']))
land_area = dict(zip(land['地块名称'], land['面积']))

# ---------- 附件1 作物 ----------
crops = pd.ExcelFile(ATT1).parse('乡村种植的农作物')
crops = crops[pd.to_numeric(crops['作物编号'], errors='coerce').notna()]
crops['作物编号'] = crops['作物编号'].astype(int)
crop_id = dict(zip(crops['作物编号'], crops['作物名称'].str.strip()))
crop_type = dict(zip(crops['作物编号'], crops['作物类型'].str.strip()))

# ---------- 附件2 2023种植 ----------
plant23 = pd.ExcelFile(ATT2).parse('2023年的农作物种植情况')
plant23['种植地块'] = plant23['种植地块'].ffill()   # 合并单元格，地块名前向填充
plant23['地块'] = plant23['种植地块'].str.strip()
plant23['作物编号'] = plant23['作物编号'].astype(int)
plant23['面积'] = plant23['种植面积/亩'].astype(float)
plant23['季次'] = plant23['种植季次'].str.strip()
plant23['季次'] = plant23['季次'].replace({'单季':'第一季'})

# ---------- 附件2 统计 ----------
stat = pd.ExcelFile(ATT2).parse('2023年统计的相关数据')
stat = stat.dropna(subset=['作物编号','地块类型'])
stat['地块类型'] = stat['地块类型'].str.strip()
stat['季次'] = stat['种植季次'].str.strip().replace({'单季':'第一季'})
stat['亩产量'] = stat['亩产量/斤'].astype(float)
stat['成本'] = stat['种植成本/(元/亩)'].astype(float)
stat['价格低'] = stat['销售单价/(元/斤)'].str.split('-').str[0].astype(float)
stat['价格高'] = stat['销售单价/(元/斤)'].str.split('-').str[1].astype(float)
stat['价格'] = (stat['价格低']+stat['价格高'])/2

# yield/cost by (crop, landtype, season)
stat['key'] = list(zip(stat['作物编号'].astype(int), stat['地块类型'], stat['季次']))
yield_map = dict(zip(stat['key'], stat['亩产量']))
cost_map  = dict(zip(stat['key'], stat['成本']))
price_map = dict(zip(stat['作物编号'].astype(int), stat['价格']))
price_lo  = dict(zip(stat['作物编号'].astype(int), stat['价格低']))
price_hi  = dict(zip(stat['作物编号'].astype(int), stat['价格高']))

# 智慧大棚第一季与普通大棚相同（附件说明）
for cid in range(17,35):
    key_put = (cid, '智慧大棚', '第一季'); key_ord = (cid, '普通大棚', '第一季')
    if key_ord in yield_map and key_put not in yield_map:
        yield_map[key_put]=yield_map[key_ord]; cost_map[key_put]=cost_map[key_ord]

# ---------- 2023 每作物每季产量 = 预期销售量基准 ----------
s = plant23.merge(land[['地块名称','地块类型']].rename(columns={'地块名称':'地块'}), on='地块', how='left')
s['类型'] = s['地块类型'].apply(lambda x: str(x).strip())
rows=[]
for _,r in s.iterrows():
    key=(int(r['作物编号']), r['类型'], r['季次'])
    y = yield_map.get(key, np.nan)
    rows.append((r['作物编号'], crop_id[int(r['作物编号'])], r['类型'], r['季次'], r['面积'], y, r['面积']*y))
prod23 = pd.DataFrame(rows, columns=['作物编号','作物名称','地块类型','季次','面积','亩产量','产量'])
prod23 = prod23.dropna(subset=['亩产量'])
prod23.to_csv('prod23.csv', index=False)

# 每个 (作物,季次) 的预期销售量
S = prod23.groupby(['作物编号','作物名称','季次'])['产量'].sum().reset_index()
S['面积2023'] = prod23.groupby(['作物编号','季次'])['面积'].sum().values
S.to_csv('sales_volume.csv', index=False)

# 每地块类型的总面积
land_sum = land.groupby('地块类型')['面积'].sum()
print('== 地块类型总面积 =='); print(land_sum)
print('\n== 每(作物,季次) 2023产量(斤, =预期销售量) 与 2023面积 ==')
print(S.to_string(index=False))
print('\n== 作物类型 ==')
print(sorted(set(crop_type.values())))
print('\n豆类作物编号:', [k for k,v in crop_type.items() if '豆类' in v])
print('\n== 求解器检查 ==')
try:
    from scipy.optimize import milp; print('scipy.optimize.milp OK (HiGHS)')
except Exception as e: print('scipy milp FAIL', e)
try:
    import pulp; print('pulp', pulp.__name__); print('cbc:', pulp.PULP_CBC_CMD().available())
except Exception as e: print('pulp FAIL', e)
