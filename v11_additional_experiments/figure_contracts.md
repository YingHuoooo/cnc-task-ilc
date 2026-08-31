# 新增实验论文图合同

绘图后端固定为 **Python/Matplotlib**，所有预览、SVG、PDF、TIFF 和 PNG 均由同一后端生成。统一导出为可编辑 SVG、字体嵌入 PDF、600 dpi TIFF 和 300 dpi PNG；目标宽度为双栏 183 mm，背景为白色。

## Figure 1：Matched ablation

- **核心结论**：在完全匹配权重、信赖域、回滚、约束、样条、优化器、学习率和两区预算后，分别移除残余时延对齐或一种锚点信息会怎样改变 V11 的任务 AUC。
- **证据链**：三个面板分别对应基线、+4 时延和三重压力；每个比较显示 V11 Full 相对消融配置的配对中位改善和 hierarchical-bootstrap 95% CI。
- **图型**：quantitative grid，+4 时延面板为主证据。
- **风险**：`task_top2` 和 `raw_top2` 是“只用一种信息源选两个区”，不是单一区预算；区间跨零必须写为未分辨，不能解释为等价。

## Figure 2：Domain-aware statistics

- **核心结论**：V11 相对无残余时延对齐的方向，在逐对、按域和分层 bootstrap 下是否一致，以及更保守重采样使区间变宽多少。
- **证据链**：每个场景并列三种 CI；原点为同一个配对效应中位数。
- **图型**：asymmetric quantitative comparison。
- **风险**：domain 数而非 task-domain pair 数是更关键的独立层级；不能用逐行 CI 代替域级不确定性。

## Figure 3：Parameter sensitivity

- **核心结论**：γ=0.25 是否位于一个宽的稳定区间，而不是唯一有效点；平滑窗口、样条数量和学习率的小范围变化是否改变结论方向。
- **证据链**：四面板分别为 γ、平滑窗口、名义控制点数和学习率；显示相对本组参考配置的配对中位 AUC 改善及 95% CI。
- **图型**：quantitative grid，γ 面板为 hero panel。
- **风险**：这是开发域一维敏感性，不是联合网格寻优；不能用结果重新选择正式 V11 参数。

## Figure 4：Representative replay

- **核心结论**：聚合 AUC 差异在一个按预设规则选出的中位案例中，如何表现为二维轨迹、逐点轮廓误差、残余时延应用和锚点历史。
- **证据链**：二维轨迹 → 轮廓误差 → 应用时延 → 选区历史。
- **图型**：asymmetric mixed-modality figure，二维轨迹为主面板。
- **风险**：案例按“最接近中位效应”自动选择，不代表最优或最差；回放是独立版本化复现实验，不冒充冻结 V11 原始点迹。

## Figure 5：Virtual plant family

- **核心结论**：在 24 个 held-out LHS 虚拟对象中，V11 相对无对齐方法的 plant-level 中位收益是否由多数对象共同支持，以及 6 个边界对象在哪里失效。
- **证据链**：三个面板对应基线、+4 时延、三重压力；每个点是一个 plant 内 5 个 demand-conflict 任务的中位改善。
- **图型**：plant-wise forest/scatter。
- **风险**：LHS 范围是预定义数值不确定性族，不代表实际机床总体；challenge 对象仅用于边界描述，不用于调参或主 claim。

## 统计与源数据合同

- 中心：成对归一化任务 AUC 改善百分比的中位数。
- 区间：固定随机种子的 20,000 次 bootstrap 95% percentile interval。
- 配对单位：同一任务、同一虚拟对象、同一场景和同一噪声实现下的方法对。
- domain bootstrap：以虚拟对象为重采样单位，保留对象内全部任务。
- hierarchical bootstrap：先重采样对象，再在对象内重采样任务，始终保持方法配对。
- 所有定量面板必须能追溯至 `results/` 中 CSV；不删除失败运行。

