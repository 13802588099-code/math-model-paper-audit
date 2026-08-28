# -*- coding: utf-8 -*-
"""全论文统一绘图样式：SimSun 中文字体（macOS 用 Songti SC 代替）、统一调色板、DPI≥300。"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# 中文字体注册：优先 SimSun，否则 macOS 的 Songti SC
CJK_FAMILY = None
for cand in ['SimSun', 'Songti SC', 'STSong', 'Noto Serif CJK SC']:
    try:
        if any(f.name == cand for f in fm.fontManager.ttflist):
            CJK_FAMILY = cand
            break
    except Exception:
        pass
if CJK_FAMILY is None:
    CJK_FAMILY = 'Songti SC'

plt.rcParams['font.sans-serif'] = [CJK_FAMILY, 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# 统一调色板
PALETTE = ['#2E5A87', '#E08E45', '#5A9E6F', '#B85C5C', '#8E6FAE', '#4E9A9A',
           '#C9A227', '#7F8C8D', '#A65E9A', '#6B8E23']
sns.set_palette(PALETTE)
sns.set_theme(style='whitegrid', font=CJK_FAMILY, palette=PALETTE)

def save(fig, path, w=None, h=None):
    if w: fig.set_size_inches(w, h if h else w * 0.72)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  ✓', path)
