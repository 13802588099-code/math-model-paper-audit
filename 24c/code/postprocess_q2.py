# -*- coding: utf-8 -*-
"""后处理：从 plan_q2_l0.3.json 写出 result2.xlsx（运行中的求解进程不含此步骤）。"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
sys.path.insert(0, 'code')
from write_results import write_result

if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'q2'
    lam = sys.argv[2] if len(sys.argv) > 2 else '0.3'
    src = f'out/plan_{tag}_l{lam}.json'
    if not os.path.exists(src):
        print(f'[跳过] 缺 {src}'); sys.exit(1)
    d = json.load(open(src))
    plant = {}
    for k, c in d['plant'].items():
        yr, s, p = k.split('|')
        plant[(int(yr), s, p)] = c
    write_result(plant, 'result2.xlsx')
