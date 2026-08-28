# -*- coding: utf-8 -*-
"""从 plan_*.json 恢复种植方案，写入 result 模板 xlsx 并校验。"""
import warnings; warnings.filterwarnings('ignore')
import sys, json
sys.path.insert(0, 'code')
import crops_data as cd
from write_results import write_result
from validate_plan import validate

def plan_from_json(path):
    d = json.load(open(path))
    plant = {}
    for k, c in d['plant'].items():
        yr, season, p = k.split('|')
        plant[(int(yr), season, p)] = c
    return plant, d

if __name__ == '__main__':
    mapping = {'out/plan_q1_1.json': 'result1_1.xlsx',
               'out/plan_q1_2.json': 'result1_2.xlsx'}
    for src, dst in mapping.items():
        plant, d = plan_from_json(src)
        print(f'== {src}: 收益 {d["profit"]:,.0f} 元')
        write_result(plant, dst)
        ok, issues = validate(plant, verbose=False)
        print('  校验:', '全部约束满足' if ok else f'{len(issues)}项违规')
        for it in issues[:8]:
            print('   -', it)
