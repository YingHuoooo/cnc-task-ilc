# 五项新增实验的冻结设计

## 共同设置

- 二维轨迹：5 类；任务制度：neutral、demand-aligned、demand-conflict；每条任务 6 个反馈无关语义区。
- 采样：161 点、6 s；4 次更新、5 次完整试次。
- 默认低维表示：每轴 12 个名义三次 B 样条基，端点钳位后每轴 10 个活动系数，共 20 个变量。
- 默认 V11：双锚点、balanced weights、0.30 全局保护权重、5.0 区域增益、学习率 0.65、信赖域/回滚、位置/速度/加速度约束、残余时延中位聚合和 γ=0.25 分数步对齐。
- 场景：基线、附加 +4 采样时延、0.05 mm 噪声 +4 时延 +1.70× 失配。
- 本轮仅运行数值虚拟 CNC 机床模型，不调用 LinuxCNC，不连接真实机床。

## 1. Matched ablation

采用 8 个全新域种子（24001–24008），覆盖 15 个任务、3 个场景和 5 种配置，共 1,800 次方法运行。

| 配置 | 唯一改变 |
|---|---|
| V11 Full | 双锚点 + γ=0.25 残余时延对齐 |
| No residual alignment | 保留双锚点，其余相同，γ=0 |
| Task-top2 | 只用公差归一化任务紧迫度排序选择前两区 |
| Raw-top2 | 只用原始轮廓误差峰排序选择前两区 |
| Uniform full trajectory | 保留同一安全/优化框架，使用全轨迹均匀权重 |

所有稀疏方法均使用两个不同活动区和 balanced weights，因此 task/raw 锚点比较不再混入权重、安全机制或预算差异。

## 2. Domain-aware statistics

对 matched ablation 和 held-out plant family 同时报告逐对、domain-level 和 hierarchical bootstrap。每种方法 20,000 次重采样，随机种子 20260820。另报告 leave-one-domain-out 中位效应范围，检查单一对象主导性。

## 3. Parameter sensitivity

采用 6 个独立开发域种子（25001–25006）、5 个 demand-conflict 任务和 +4/三重压力场景。一维改变参数，不做联合寻优：

- γ：0、0.10、0.25、0.40、0.60、1.00；
- 速度相关平滑窗口：3、5、7、9；
- 每轴名义样条控制点：8、12、16；
- 学习率：0.50、0.65、0.80。

共 780 次方法运行。该实验只描述稳定区间，不改变 V11 的冻结默认参数。

## 4. Representative replay

在优先级 1 的 +4 时延、demand-conflict、V11 Full vs No residual alignment 配对中，自动选择改善百分比最接近总体中位数的案例。用相同对象、任务、场景和噪声种子独立重放，保存全部命令、反馈、轮廓误差、时延与锚点历史。

## 5. Virtual plant family

- 主验证：24 个由 14 维 Latin Hypercube Sampling 生成、此前未见的对象；
- 边界验证：6 个显式 challenge 对象，分别强调最大时延、低带宽、高轴不对称、强耦合、强摩擦和紧饱和；
- 任务：5 个 demand-conflict；场景：3 个；方法：V11 Full、无对齐、Raw-top2、Uniform；
- 共 1,800 次方法运行。

LHS 对象用于主分析；challenge 对象不用于重新选择 γ 或任何方法参数。

