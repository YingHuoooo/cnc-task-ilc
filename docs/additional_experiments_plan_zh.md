# V11 论文后续工作计划（纯仿真路线）

> 基于当前论文 `complete_sci_manuscript_v11.pdf` 的方法、局限性、TODO，以及本轮讨论整理。  
> 目标：**不继续扩展“V8→V9→V10→V11”的版本史，而是围绕最终方法 V11，补齐一篇纯仿真方法论文最关键的验证证据。**
>
> 核心原则：**增加独立证据，而不是单纯增加运行次数。**

---

## 0. 总体定位

最终论文应以 **V11 作为唯一 proposed method / final technical route**：

**Task-aware dual-anchor iterative trajectory learning + residual-delay alignment**

论文主体不再强调 V8、V9、V10、V12、V13 是“不同代算法”，而把它们重新组织为：

- 基础模块验证；
- 失效因素诊断；
- 最终方法确认；
- 边界研究。

后续仿真工作做到下面 **5 个优先级** 后即可收口投稿，不建议继续无限扩展 V14/V15。

---

# 优先级 1：做 V11 的 matched ablation（模块消融）

## 目的

回答审稿人最自然的问题：

> V11 的性能提升到底来自哪些模块？

当前论文中早期 dual-anchor 与 error-peak 的比较还混有 weighting、trust/rollback 等配置差异，因此不适合作为“dual-anchor 本身的独立因果证据”。

后续应改成标准的 **V11-centered ablation**。

## 推荐比较组

尽量保持以下内容一致：

- weighting；
- trust radius；
- rollback；
- constraints；
- spline basis；
- optimizer；
- learning rate；
- trial budget。

只改变被研究模块。

建议至少包含：

1. **V11 Full**
   - task anchor
   - raw-error anchor
   - residual-delay alignment

2. **V11 w/o residual-delay alignment**
   - task anchor
   - raw-error anchor
   - 不做 sensitivity phase alignment

3. **Task-anchor only**
   - 仅保留 task urgency anchor

4. **Raw-error only / matched error-driven**
   - 仅保留 raw geometric error anchor

如计算成本允许，可再增加：

5. **Uniform/full-trajectory matched baseline**

## 核心问题

需要回答：

- Task anchor 是否有独立贡献？
- Raw-error anchor 是否有独立贡献？
- 两者组合是否互补？
- Residual-delay alignment 是否在同一套基础框架下提供额外收益？

## 输出建议

- median improvement
- paired win rate
- 95% CI
- forest plot
- component-wise ablation table

---

# 优先级 2：升级统计分析为 domain-aware / hierarchical statistics

## 目的

目前大量 task 共享同一个 virtual plant/domain。

因此：

> “task-domain pair 数很多”不等于“独立样本很多”。

后续统计应把 **plant/domain** 视为更重要的独立层级。

## 建议同时报告

### A. 当前 paired bootstrap

作为与已有结果兼容的统计结果。

### B. Domain-level bootstrap

以 virtual plant/domain 为重采样单位。

### C. Hierarchical bootstrap

推荐层级：

```text
domain
└── task
    └── paired method result
```

即：

1. 先重采样 domain；
2. 再在该 domain 内重采样 tasks；
3. 保留方法之间的 paired structure。

## 需要观察

重点不是“必须显著”，而是：

- V11 主要结论在更保守统计假设下是否仍成立；
- CI 是否明显变宽；
- 是否存在某几个 domain 主导全部收益。

## 输出建议

同一个结果表中并列：

| Effect | Paired Bootstrap | Domain Bootstrap | Hierarchical Bootstrap | Win Rate |
| ------ | ---------------: | ---------------: | ---------------------: | -------: |

如果主要结论在三种统计方法下方向一致，论文可信度会明显提高。

---

# 优先级 3：做核心参数敏感性分析

## 目的

证明：

> V11 不是因为某一个“碰巧选对的超参数”才有效。

不需要做巨大的 grid search。

目标是证明主要结果在一个合理参数区间内具有稳定性。

## 第一优先参数：Residual-delay shrinkage

当前核心参数：

```text
γ = 0.25
```

建议测试：

```text
γ ∈ {0, 0.10, 0.25, 0.40, 0.60, 1.00}
```

重点观察：

- γ = 0：相当于没有 delay alignment；
- γ 较小：保守补偿；
- γ = 1：完全相信估计值，是否出现过补偿；
- 是否存在一个较宽的稳定区域，而不是只有 0.25 一个点有效。

### 理想结果形式

如果出现类似：

```text
0      -> delay 条件下明显下降
0.1    -> 开始改善
0.25   -> 稳定
0.4    -> 仍然稳定
0.6    -> 波动变大
1.0    -> 容易过补偿
```

那么 `γ=0.25` 就从 heuristic parameter 变成有机制依据的 conservative design。

## 第二层可选参数

建议从以下参数中选 2–3 个，不要全部扫：

- semantic-zone width；
- active-zone weight；
- background weight；
- spline basis 数量；
- smoothing window；
- lag search range；
- learning rate；
- trust radius。

## 输出建议

- sensitivity curve；
- boxplot / violin / interval plot；
- 结论强调“稳定区间”，而不是寻找“最优点”。

---

# 优先级 4：补 representative trajectory replay

## 目的

当前论文 aggregate metrics 很完整，但缺少让读者一眼看懂的轨迹级证据。

需要回答：

> AUC 改善 4%–7% 到底在轨迹上表现成什么？

## 注意

当前 frozen summary CSV 不包含所有正式实验的完整 pointwise best-command / feedback traces。

因此：

**不能从 summary data 人工拼接或重建轨迹。**

应做：

> independently versioned representative replay

并保存完整 pointwise trace。

## 推荐案例

选择一个典型但不是人为挑极端的 case，例如：

- demand-conflict task；
- +4 delay；
- 或 combined stress；
- uncompensated 方法确实受 delay 影响；
- V11 有明显但不过度夸张的改善。

## 推荐图 1：二维轨迹

叠加：

- Reference contour
- Initial response
- Final uncompensated dual-anchor
- Final V11
- 6 个 semantic zones

目标：

让读者看到关键区域的几何改善。

## 推荐图 2：Contour error vs sample index

绘制：

- initial
- intermediate trials（可选）
- final uncompensated
- final V11
- semantic zones
- zone tolerance / normalized boundary（如适合）

## 推荐图 3：Residual lag evolution

显示：

- x-axis estimated residual lag
- y-axis estimated residual lag
- filtered/cumulative estimate
- applied fractional shift

## 推荐图 4（可选）：Anchor history

显示每次 update：

```text
Trial 0 -> task anchor = Z?
          raw anchor  = Z?

Trial 1 -> ...
```

这样可以直观看出 task-aware allocation 的行为。

---

# 优先级 5：扩大 virtual plant family

这是纯仿真路线中最重要的泛化升级。

## 核心原则

不要把：

```text
6000 runs
```

简单扩成：

```text
20000 runs
```

更重要的是增加：

> **independently sampled virtual plants**

即增加“独立系统数量”，而不是只增加同一系统上的重复次数。

---

## 5.1 建立明确的 plant uncertainty family

保持当前 plant structure，不必重新发明 simulator。

当前框架已经包含：

- axis-specific second-order dynamics；
- axis-dependent delay；
- nonlinear friction；
- cross-axis coupling；
- saturation；
- repeatable disturbance；
- measurement noise / extra delay / mismatch stress。

后续主要随机化参数。

## 推荐随机参数

### Axis dynamics

- `ωx`, `ωy`
- `ζx`, `ζy`
- x/y bandwidth asymmetry

### Delay

- `dx`, `dy`
- axis-specific integer/effective lag

### Coupling

- `cxy`
- `cyx`

### Friction

- friction magnitude
- friction asymmetry

### Saturation

- command / velocity saturation level

### Disturbance

- amplitude
- frequency / spectral composition
- axis dependence

### 可选

- sensor noise level
- dynamic mismatch scale
- low-frequency drift（若仍保持 stationary plant，则不要加入长期漂移）

---

## 5.2 采样方式

不建议完全随意 uniform random。

更推荐：

- Latin Hypercube Sampling (LHS)
- Sobol sequence

原因：

> 能更均匀覆盖多维参数空间，减少 plant 都挤在某一小块区域的问题。

---

## 5.3 Plant 数量建议

### 最低可接受

```text
20–30 independent plants
```

### 推荐

```text
30–50 independent plants
```

### 如果计算资源充足

```text
50–100 independent plants
```

但超过 50 后，收益通常不如增加“覆盖质量”和“统计设计”。

---

## 5.4 不要重新跑整个版本历史

新增 large plant family 后，不建议重跑：

```text
V8 + V9 + V10 + V11 + V12 + V13
```

只保留与最终论文结论直接相关的方法。

## 推荐最终比较

最多 3–4 个方法：

1. **V11 Full**
2. **V11 w/o residual alignment**
3. **Matched error-driven baseline**
4. **Full-trajectory baseline（可选）**

## 推荐场景

不需要继续七八种 scene。

可压缩为：

1. **Base / nominal-domain condition**
2. **Delay-dominated condition**
3. **Combined stress condition**

这样就能回答最终问题：

> 当 plant dynamics 广泛变化以后，V11 的主要优势是否仍然存在？

---

# 5.5 In-distribution 与 Edge / Challenge plants

建议把 plant family 分成两类。

## A. In-distribution family

用于正式主 claim。

参数范围应代表预先定义的合理 virtual CNC feed-system uncertainty。

回答：

> V11 在目标 plant family 内是否稳定有效？

## B. Edge / challenge plants

取 uncertainty region 的边界，例如：

- 最大 delay；
- 最大 x/y bandwidth asymmetry；
- 强 coupling；
- 强 friction；
- 较低 bandwidth；
- saturation 较紧。

这些数据：

- 不用于重新 tuning；
- 不用于选择 γ；
- 只做 robustness boundary analysis。

回答：

> V11 在什么条件附近开始失效？

---

# 5.6 最重要的统计单位：Plant

最终大规模实验应把 plant 作为主要展示单位之一。

例如：

```text
Across 40 independently sampled virtual plants,
V11 outperformed the uncompensated method in 34/40 plants...
```

并报告：

- plant-level median improvement；
- plant-level 95% CI；
- plant win rate；
- task-within-plant distribution。

## 推荐图

### Plant-wise forest/scatter

纵轴：

```text
Plant 01
Plant 02
...
Plant 40
```

横轴：

```text
V11 improvement over baseline (%)
```

加一条 0% vertical line。

这张图可以直观看出：

- 是否多数 plant 都受益；
- 是否少数 plant 拉高总体平均；
- 哪些 plant 是 failure cases。

---

# 5.7 建议冻结开发与正式验证

如果条件允许，采用更严格的 protocol。

## Development plants

例如：

```text
10–15 plants
```

用于：

- 参数选择；
- sensitivity study；
- 方法调试；
- replay case 设计。

## Held-out formal plants

例如：

```text
30–40 plants
```

一旦 V11 和所有参数冻结后再运行。

正式 plants：

- 不用于调 γ；
- 不用于调 zone weight；
- 不用于挑 optimizer 参数；
- 不用于修改算法。

这样可以明显增强“泛化验证”的说服力。

---

