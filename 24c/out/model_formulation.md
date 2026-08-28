# 模型公式草稿（将写入论文 5MakeModel）

## 问题1：确定性 MILP

### 集合
- $\mathcal{P}$ 地块集合（54块）：$\mathcal{P}_G$（平旱地/梯田/山坡地，单季粮食），$\mathcal{P}_D$（水浇地），$\mathcal{P}_E$（普通大棚），$\mathcal{P}_F$（智慧大棚）
- $\mathcal{T}=\{2024,\dots,2030\}$ 年份
- $\mathcal{S}\in\{\text{第一季},\text{第二季}\}$ 种植季次
- $\mathcal{C}$ 作物集合；$\mathcal{C}^{b}$ 豆类作物；$\mathcal{C}_{ps}$ 地块$p$季次$s$可种作物
- 销售单位 $\mathcal{K}=\{(c,s): \text{作物}c\text{季次}s\text{有2023年销售基准}\}$

### 参数
- $A_p$：地块面积（亩）
- $y_{c,p,s}$：作物$c$在类型为$t(p)$地块季次$s$的亩产量（斤/亩）
- $p^{r}_{c,s}$：销售单价（元/斤，取2023区间中点）
- $c_{c,p,s}$：种植成本（元/亩）
- $S_{c,s}$：预期销售量 = 2023年产量（斤）
- $\delta$：最小种植面积比例（本文采用单作物整地块，$\delta=1$）

### 决策变量
$$x_{p,t,s,c}=\begin{cases}1 & \text{地块}p\text{第}t\text{年第}s\text{季种植作物}c\\0 & \text{否则}\end{cases}$$

### 目标（情形1，滞销浪费）
$$\max \sum_{p,t,s,c} \left(p^{r}_{c,s} y_{c,p,s} - c_{c,p,s}\right) A_p x_{p,t,s,c}$$

### 约束
1. 土地容量：$\sum_{c\in\mathcal{C}_{ps}} x_{p,t,s,c} \le 1$
2. 季节适应性（可种作物集合限定，见$\mathcal{C}_{ps}$）
3. 水浇地结构：$\sum_{c\in\{35,36,37\}} x_{p,t,\text{二},c} + x_{p,t,\text{一},\text{水稻}} = 1$
4. 重茬约束：$x_{p,t,s,c} + x_{p,t+1,s,c} \le 1$
5. 豆类轮作：$\forall p, \forall w\in\{2023,\dots,2028\}: \sum_{t\in\{w,w+1,w+2\}}\sum_{s}\sum_{c\in\mathcal{C}^{b}} x_{p,t,s,c} \ge 1$（$w=2023$时并入2023已知种植）
6. 销售上限：$\sum_p A_p y_{c,p,s} x_{p,t,s,c} \le S_{c,s}$

### 情形2（超产50%降价）
引入 $\mu_{c,s,t}$（原价售出量）、$\nu_{c,s,t}$（降价售出量）：
$$\mu_{c,s,t}+\nu_{c,s,t}=\sum_p A_p y_{c,p,s}x_{p,t,s,c},\quad \mu_{c,s,t}\le S_{c,s}$$
$$\max \sum_{c,s,t}\left(p^{r}_{c,s}\mu_{c,s,t}+0.5 p^{r}_{c,s}\nu_{c,s,t}\right)-\sum_{p,t,s,c}c_{c,p,s}A_p x_{p,t,s,c}$$

## 问题2：两阶段随机规划 + CVaR

- 随机参数（情景$\omega\in\Omega$）：
  - 销售量 $\tilde S_{c,s,t}(\omega)$：小麦/玉米年增5-10%，其他$\pm5\%$
  - 亩产量 $\tilde y(\omega)\sim(1\pm10\%)y$
  - 价格：粮食稳、蔬菜$+5\%$/年、食用菌$-1\sim-5\%$/年
  - 成本 $+5\%$/年（确定性）
- 第一阶段决策 $x$；第二阶段 $\mu_\omega,\nu_\omega$
- 目标：$\max \mathbb{E}_\omega[\Pi_\omega(x)] - \lambda \mathrm{CVaR}_\alpha(\mathcal{L}_\omega)$
- $\mathrm{CVaR}_\alpha = \eta + \frac{1}{(1-\alpha)|\Omega|}\sum_\omega \zeta_\omega,\ \zeta_\omega\ge -\Pi_\omega-\eta,\ \zeta_\omega\ge0$

## 问题3：相关随机模拟

- 相关结构：替代组（负相关-0.6）、互补组（正相关+0.4）、销量-价格（-0.5）、产量-价格（-0.3）、产量-销量（+0.2）
- Cholesky 分解 $\Sigma=LL^\top$，$z=L\xi$ 生成相关标准正态
- 映射到销量/产量/价格乘子，代入随机规划求解
- 与问题2（独立情景）对比
