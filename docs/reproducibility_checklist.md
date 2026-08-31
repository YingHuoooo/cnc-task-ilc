# 06 复现、项目管理与完成检查

## 1. 推荐代码目录

未来开始编码时，建议在本研究包旁建立：

~~~text
cnc_task_ilc/
    README.md
    pyproject.toml
    environment.yml
    LICENSE
    configs/
        base.yaml
        machines/
        trajectories/
        methods/
        experiments/
    linuxcnc/
        configs/
        hal/
        gcode/
        components/
    src/
        controller_interface/
        virtual_plant/
        trajectory/
        task_state/
        window_detection/
        nominal_model/
        residual_model/
        optimizer/
        logging/
        evaluation/
    scripts/
        run_single_trial.py
        run_ilc_episode.py
        run_experiment.py
        aggregate_results.py
        make_paper_figures.py
    tests/
        test_trajectory_alignment.py
        test_contour_error.py
        test_jacobian.py
        test_qp_constraints.py
        test_rollback.py
        test_data_split.py
    data/
        raw/
        processed/
        splits/
    runs/
    results/
        tables/
        figures/
        statistics/
    paper/
        manuscript.tex
        references.bib
        figures/
        supplementary/
~~~

## 2. 配置优先

所有实验参数应来自 YAML/JSON，不应散落在代码中。配置至少包含：

- software；
- trajectory；
- LinuxCNC；
- learner model；
- virtual plant；
- critical windows；
- optimizer；
- residual model；
- trials；
- split；
- random seeds；
- logging；
- output。

模板见 [project_config_template.yaml](project_config_template.yaml)。

## 3. 版本和运行记录

每次运行保存：

- Git commit；
- dirty worktree 标记；
- LinuxCNC 版本；
- Python 版本；
- OS 和内核；
- 优化器版本；
- 配置文件哈希；
- 随机种子；
- 开始/结束时间；
- 主机信息；
- 运行状态和异常。

## 4. 单元测试

### 4.1 轨迹和任务状态

- 直线上的法向误差符号正确；
- 圆轨迹上轮廓误差与径向误差一致；
- 时间/相位重采样不改变几何单位；
- 导数滤波不会引入明显时间错位；
- 窗口索引与原始采样点一致。

### 4.2 名义模型和雅可比

- 阶跃响应符合设定二阶参数；
- 离散模型稳定；
- 解析/自动微分雅可比与中心有限差分一致；
- 有限差分步长变化不导致方向翻转；
- 关键窗口切片维度正确。

### 4.3 QP

- 零误差时输出接近零更新；
- 无约束情况下与线性最小二乘一致；
- 速度、加速度和跃度约束均满足；
- 不可行时返回明确状态；
- 松弛变量和违反量被记录；
- 信赖域缩放方向正确。

### 4.4 ILC 状态机

- 接受更新后 trial index 增加；
- 拒绝更新后恢复上一个已接受参数；
- 最终结果是最佳已接受试次；
- 达到停止条件后不再运行；
- 异常中断不会覆盖已有日志。

### 4.5 数据隔离

- Test 域参数不能出现在训练配置；
- 测试轨迹哈希与训练轨迹不同；
- 标准化统计只来自训练集；
- 残差模型在正式测试前冻结；
- 结果聚合不丢弃失败运行。

## 5. 梯度和优化数值检查

对随机选择的控制参数方向 \(d\) 检查：

\[
\frac{
J(\kappa+\epsilon d)-J(\kappa-\epsilon d)}
{2\epsilon}
\approx
\nabla_\kappa J^\top d
\]

至少在：

- 理想名义模型；
- 轻度失配；
- 关键窗口边界；
- 接近约束；
- 不同样条维度

下执行检查。

记录相对误差：

\[
\epsilon_{\mathrm{grad}}
=
\frac{|g_{\mathrm{fd}}-g_{\mathrm{model}}|}
{\max(1,|g_{\mathrm{fd}}|,|g_{\mathrm{model}}|)}
\]

阈值应根据数值精度和离散模型确定，并在测试中固定。

## 6. 实验注册表

每次正式运行应在注册表中包含：

| 字段 | 说明 |
|---|---|
| run_id | 唯一运行编号 |
| method_id | 方法版本 |
| trajectory_id | 轨迹编号 |
| machine_domain | 虚拟机床域 |
| noise_seed | 噪声种子 |
| code_commit | 代码版本 |
| config_hash | 配置哈希 |
| status | success/failure/timeout |
| trial_count | 实际试次数 |
| result_path | 结果位置 |

聚合脚本必须根据注册表读取运行，禁止通过手工复制文件选择“好结果”。

## 7. 项目阶段和交付物

### Stage A：可观测仿真环境

交付：

- LinuxCNC 可运行配置；
- 三条基础 G-code；
- 命令/反馈/轮廓误差日志；
- 理想回环和虚拟轴响应图。

完成条件：

- 能稳定批量运行；
- 反馈与命令不再理想相等；
- 数据时间对齐通过测试。

### Stage B：标准受约束 ILC

交付：

- 样条参数化；
- 名义模型；
- 雅可比；
- QP；
- 回退；
- 标准 ILC 学习曲线。

完成条件：

- 至少在简单域中降低误差；
- 梯度和约束测试通过；
- 失败不会破坏运行状态。

### Stage C：关键窗口方法

交付：

- 固定窗口基线；
- 自动窗口检测；
- 窗口可视化；
- 相同窗口预算对比。

完成条件：

- 自动窗口不只是复制曲率阈值；
- 至少一个非几何关键事件得到合理识别；
- 自动窗口消融可运行。

### Stage D：模型失配安全性

交付：

- 参数和结构失配域；
- 信赖域；
- 预测/实际改善比；
- 不确定度或残差；
- 回退统计。

完成条件：

- 正式定义发散和恶化更新；
- 安全模块有明确测量；
- 测试域参数对学习器隐藏。

### Stage E：正式实验

交付：

- 冻结配置；
- 主比较；
- 消融；
- OOD；
- 失败案例；
- 计算成本；
- 统计报告。

### Stage F：论文

交付：

- 主文；
- 附录/补充材料；
- 图表；
- BibTeX；
- 代码和数据说明；
- claim-evidence 审计；
- 审稿风险自查。

## 8. 最小完成版本

如果项目只作为博士小论文，以下内容完成即可形成闭环：

- 二轴，不扩展五轴；
- 五类轨迹；
- 多个虚拟机床参数域；
- 全轨迹 ILC、固定窗口 ILC 和自动窗口 ILC；
- 约束 QP 和回退；
- 至少一个模型失配实验；
- 至少一个未见域实验；
- 主比较、核心消融、失败案例；
- 可重复运行脚本。

残差深度网络、元学习和复杂切削过程均为可选，不是最小版本条件。

## 9. 决策门

### Gate 1：标准 ILC 是否能在简单域工作？

- 否：优先修正时间对齐、误差符号、雅可比和更新方向；
- 是：进入窗口研究。

### Gate 2：固定窗口是否优于全轨迹？

- 否：检查任务是否真正具有稀疏关键事件；
- 是：自动窗口具有研究基础。

### Gate 3：自动窗口是否优于固定规则？

- 否：论文可转向模型失配安全更新；
- 是：以自动事件发现为主要贡献。

### Gate 4：不确定度是否优于固定小步长？

- 否：删除不必要的学习模型，保留简单信赖域；
- 是：保留为第二贡献。

### Gate 5：未见域是否有优势？

- 否：收缩为域内自适应论文；
- 是：可合理声称 sim-to-sim 泛化。

## 10. 数据和图形质量

- 保存原始时间序列，不只保存聚合指标；
- 图形由脚本生成；
- 图表标题写清数据集、域、试次和统计；
- 单位一致；
- 误差方向明确；
- 最优和次优标记规则固定；
- 失败样本进入统计；
- 不截断纵轴制造夸大效果；
- 置信区间计算单位是独立轨迹—域组合，不是时间采样点。

## 11. 论文内容审计

正式投稿前逐条检查：

| 项目 | 状态 |
|---|---|
| 摘要每个数字可追溯到结果文件 | 待完成 |
| Introduction 每条现状主张有原始文献 | 待完成 |
| Method 可从文字和配置重现 | 待完成 |
| 每项贡献有消融 | 待完成 |
| 最强相关基线存在 | 待完成 |
| 测试集未用于调参 | 待完成 |
| 失败案例没有删除 | 待完成 |
| 纯仿真边界写入摘要、讨论和结论 | 待完成 |
| 代码和随机种子归档 | 待完成 |
| 图表数字与正文一致 | 待完成 |

## 12. 主要风险登记

| 风险 | 后果 | 预防/降级 |
|---|---|---|
| 虚拟 plant 太简单 | 被认为玩具问题 | 结构失配、非线性、OOD |
| 自动窗口等同曲率阈值 | 创新不足 | 非几何事件、强基线、相同预算 |
| 网络模块无明显收益 | 复杂度无净价值 | 删除网络，保留可解释评分 |
| QP 频繁不可行 | 实验中断 | 松弛变量、信赖域、约束诊断 |
| 只改善局部、恶化全局 | 结论失效 | 全局保护项和 Pareto 报告 |
| 不可重复噪声过强 | ILC 失效 | 明确边界，保留反馈控制 |
| 训练/测试泄漏 | 泛化无效 | 配置审计、哈希和冻结 |
| 没有真实机床 | 制造结论受限 | 定位为算法/仿真论文 |
| 工作量过大 | 无法形成小论文 | 按 Gate 删除元学习和切削模型 |

