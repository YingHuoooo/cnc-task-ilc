# 02 技术方法与输入输出

## 1. 术语和符号表

| 规范术语 | 英文/缩写 | 含义 |
|---|---|---|
| 任务级迭代学习控制 | Task-Level Iterative Learning Control, TL-ILC | 跨重复试次，根据任务结果更新前馈命令 |
| 关键任务窗口 | critical task window | 对最终任务指标影响较大的连续时间区间 |
| 学习器名义模型 | learner nominal model | 用于计算局部梯度的低精度模型 |
| 虚拟物理机床 | virtual physical machine | 独立生成“实际”反馈的高保真仿真对象 |
| 模型失配 | model mismatch | 名义模型与虚拟物理机床的结构或参数差异 |
| 任务状态 | task state | 用于评价加工结果的轮廓、动态或过程状态 |
| 信赖域 | trust region | 限制单轮更新幅值的局部有效范围 |
| 残差模型 | residual model | 学习名义预测与实际反馈之间的偏差 |
| 试次 | trial / iteration | 同一轨迹任务的一次完整重复执行 |

主要符号：

| 符号 | 维度/类型 | 定义 |
|---|---|---|
| \(k\) | 整数 | ILC 试次编号 |
| \(t\) | 连续/离散时间 | 任务相位或采样时刻 |
| \(r(t)\) | \(\mathbb{R}^d\) | 几何参考路径 |
| \(u_k(t)\) | \(\mathbb{R}^m\) | 第 \(k\) 次的轴前馈命令 |
| \(\kappa_k\) | \(\mathbb{R}^p\) | 低维样条或局部修正参数 |
| \(y_k(t)\) | \(\mathbb{R}^m\) | 虚拟物理机床反馈位置 |
| \(x_k(t)\) | \(\mathbb{R}^q\) | 任务状态 |
| \(e_k(t)\) | \(\mathbb{R}^q\) | 任务状态误差 |
| \(W_k\) | 时间索引集合 | 选中的关键窗口并集 |
| \(s_k(t)\) | 标量 | 时刻 \(t\) 的关键性评分 |
| \(M_k^W\) | 矩阵 | 关键窗口任务状态对 \(\kappa\) 的局部雅可比 |
| \(\Delta\kappa_k\) | \(\mathbb{R}^p\) | 本轮轨迹参数修正 |
| \(\rho_k\) | 标量 | 信赖域半径 |
| \(\sigma_k(t)\) | 标量/向量 | 残差模型不确定度 |

## 2. 总体输入输出

### 2.1 项目输入

#### A. 任务输入

- G-code 文件，或参数化二维参考路径；
- 参考进给速度；
- 采样周期；
- 轴数及坐标系定义；
- 允许的路径偏差；
- 速度、加速度、跃度限制；
- 最大 ILC 试次数。

#### B. 初始轨迹输入

- LinuxCNC 轨迹规划器输出；
- 或由普通离线轨迹优化得到的名义轨迹；
- 或由 CAM/G-code 插补得到的轴指令。

#### C. 学习器模型输入

- 二阶轴模型参数：固有频率、阻尼、静态增益；
- 可选名义时延；
- 样条基函数；
- 局部线性化方式；
- 约束矩阵与代价权重。

#### D. 虚拟物理机床输入

- 真实用于评估但不直接暴露给学习器的轴模型参数；
- 非线性摩擦、死区、饱和、时延；
- 传感器噪声和工况漂移；
- 随机种子；
- 域编号和训练/测试标记。

### 2.2 单轮算法输入

第 \(k\) 次更新接收：

\[
\mathcal{I}_k =
\{r(t),u_k(t),y_k(t),x_k(t),e_k(t),
\hat f,\mathcal C,\mathcal H_{0:k}\}
\]

其中：

- \(\hat f\)：学习器名义模型；
- \(\mathcal C\)：运动学和更新约束；
- \(\mathcal H_{0:k}\)：截至当前的试次历史。

### 2.3 单轮算法输出

\[
\mathcal{O}_k =
\{W_k,s_k(t),M_k^W,\Delta\kappa_k,
u_{k+1}(t),q_k\}
\]

其中 \(q_k\) 是本轮诊断信息，包括：

- QP 是否可行；
- 预测改善量；
- 实际改善量；
- 信赖域比值；
- 是否接受更新；
- 是否发生回退；
- 是否满足停止条件。

### 2.4 最终输出

- 最终 G-code 或轨迹修正参数；
- 每轮命令轨迹和反馈轨迹；
- 关键窗口位置及评分；
- 关键误差和全局误差曲线；
- 约束违反报告；
- 学习收敛报告；
- 未见轨迹/未见机床泛化结果；
- 可用于论文表格和图形的聚合数据。

## 3. 问题定义

### 3.1 前馈轨迹参数化

用基函数矩阵 \(B(t)\) 表示前馈轨迹：

\[
u_k(t)=u_0(t)+B(t)\kappa_k
\]

其中 \(u_0(t)\) 是 LinuxCNC 产生的名义指令，\(\kappa_k\) 只表示学习修正。这样能够：

- 保留原始 G-code 的整体几何意图；
- 降低变量维度；
- 抑制高频逆补偿；
- 方便对速度、加速度和跃度施加线性约束。

推荐初始实现：

- 每个轴独立使用三次 B 样条；
- 控制点间隔覆盖 20-100 个控制采样周期；
- 在关键窗口附近允许更密的控制点；
- 窗口之外使用稀疏控制点或正则化保持零修正。

### 3.2 任务状态

对于二维轮廓任务，参考位置为 \(r(t)\)，反馈位置为 \(y_k(t)\)。定义切向单位向量 \(\tau(t)\) 和法向单位向量 \(n(t)\)：

\[
\tau(t)=\frac{\dot r(t)}{\|\dot r(t)\|+\epsilon},
\qquad
n(t)=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}\tau(t)
\]

近似切向误差和法向轮廓误差：

\[
e_{\mathrm{tan},k}(t)=\tau(t)^\top(y_k(t)-r(t))
\]

\[
e_{\mathrm{con},k}(t)=n(t)^\top(y_k(t)-r(t))
\]

任务状态可定义为：

\[
x_k(t)=
\begin{bmatrix}
e_{\mathrm{con},k}(t)\\
e_{\mathrm{tan},k}(t)\\
\dot y_k(t)-\dot r(t)\\
\ddot y_k(t)-\ddot r(t)
\end{bmatrix}
\]

最小可发表版本以轮廓误差为主；速度和加速度误差可作为关键窗口评分特征或安全项，而不是全部作为核心目标。

### 3.3 全局和关键窗口目标

全局误差：

\[
J_{\mathrm{global},k}
=
\frac{1}{T}\sum_t e_{\mathrm{con},k}(t)^2
\]

关键窗口目标：

\[
J_{\mathrm{critical},k}
=
\sum_{t\in W_k}
w_k(t)\,
e_k(t)^\top Q e_k(t)
\]

安全目标：

\[
J_{\mathrm{safe},k}
=
\lambda_v\|D_1u_k\|^2
+\lambda_a\|D_2u_k\|^2
+\lambda_j\|D_3u_k\|^2
\]

总目标不应只优化关键窗口，否则可能将误差转移到其他区域。推荐：

\[
J_k
=
J_{\mathrm{critical},k}
+\eta J_{\mathrm{global},k}
+J_{\mathrm{safe},k}
\]

其中 \(\eta>0\) 防止全局误差恶化。

## 4. 模块一：候选关键窗口生成

### 4.1 设计动机

原论文需要人工标注绳索自碰撞时刻。数控轨迹中的重要区域不一定只对应几何拐角，也可能由轴动态差异、时延、饱和或误差传播造成，因此不能只用曲率阈值。

### 4.2 候选特征

对每个时间点计算：

- 几何曲率 \(|c(t)|\)；
- 参考速度 \(\|\dot r(t)\|\)；
- 参考加速度 \(\|\ddot r(t)\|\)；
- 参考跃度 \(\|\dddot r(t)\|\)；
- 当前轮廓误差 \(|e_{\mathrm{con},k}(t)|\)；
- 误差变化率 \(|\Delta e_{\mathrm{con},k}(t)|\)；
- 误差局部峰值；
- 局部任务灵敏度 \(\|\partial J/\partial\kappa_t\|\)；
- 残差模型不确定度 \(\sigma_k(t)\)；
- 是否接近速度、加速度或控制饱和。

### 4.3 推荐的最小算法

先对特征做稳健归一化：

\[
\tilde z_i(t)
=
\frac{z_i(t)-\operatorname{median}(z_i)}
{\operatorname{IQR}(z_i)+\epsilon}
\]

评分：

\[
s_k(t)
=
\alpha_1|\tilde e_k(t)|
+\alpha_2|\widetilde{\Delta e}_k(t)|
+\alpha_3\widetilde{\|\nabla_{\kappa_t}J\|}
+\alpha_4\tilde\sigma_k(t)
+\alpha_5\tilde c(t)
+\alpha_6\widetilde{\operatorname{sat}}(t)
\]

然后执行：

1. 一维平滑；
2. 非极大值抑制；
3. 选择前 \(K\) 个峰值；
4. 以峰值为中心扩展固定或自适应半宽；
5. 合并重叠窗口；
6. 限制窗口总长度占整条轨迹的比例。

该版本易解释、易消融、无需大规模训练，适合作为第一篇小论文。

### 4.4 计算机化增强版本

若基础算法已经稳定，可将窗口评分器替换为：

- 小型时序卷积网络；
- Transformer 编码器；
- 可微 Top-K/稀疏门控；
- 变化点检测器；
- 基于历史试次的上下文 bandit。

但必须保证：

- 有足够多的虚拟轨迹和机床域；
- 训练数据与测试域隔离；
- 与非学习评分器公平比较；
- 网络开销与收益匹配；
- 不将窗口真实标签从测试系统泄漏给算法。

## 5. 模块二：学习器名义模型与局部雅可比

### 5.1 名义轴模型

第 \(i\) 个轴可用：

\[
G_i(s)
=
\frac{K_i\omega_{n,i}^2}
{s^2+2\zeta_i\omega_{n,i}s+\omega_{n,i}^2}
e^{-\tau_i s}
\]

第一阶段可忽略名义时延，第二阶段加入近似时延。学习器不能访问虚拟物理机床的全部真实参数。

### 5.2 局部线性化

记名义模型产生的任务状态为：

\[
\hat x(\kappa)=\mathcal T(\hat f(u_0+B\kappa))
\]

在当前参数 \(\kappa_k\) 附近：

\[
\hat x(\kappa_k+\Delta\kappa)
\approx
\hat x(\kappa_k)+M_k\Delta\kappa
\]

\[
M_k
=
\left.
\frac{\partial\hat x}{\partial\kappa}
\right|_{\kappa_k}
\]

只取关键窗口行得到 \(M_k^W\)。

### 5.3 雅可比计算方式

推荐优先级：

1. 解析离散状态空间灵敏度；
2. 自动微分；
3. 中心有限差分；
4. 实验数据局部回归。

最初实现可采用中心有限差分：

\[
M_{:,j}
\approx
\frac{\hat x(\kappa_k+h e_j)-\hat x(\kappa_k-h e_j)}
{2h}
\]

需要对步长 \(h\) 做敏感性实验，避免数值噪声或非线性误差。

## 6. 模块三：残差和不确定度

### 6.1 作用

残差模型不是替代物理模型，而是预测：

\[
\delta x_k(t)
=
x_k(t)-\hat x_k(t)
\]

或者预测局部雅可比偏差：

\[
\Delta M_k=M_{\mathrm{actual},k}-M_{\mathrm{nominal},k}
\]

第一篇小论文推荐预测状态残差和不确定度，不直接预测完整雅可比，后者维度更高且训练不稳定。

### 6.2 输入特征

局部特征向量可以包含：

\[
z_k(t)=
[
u,\dot u,\ddot u,\dddot u,
\hat y,\hat{\dot y},
e_{k-1},
\Delta u_{k-1},
c(t),
s_{k-1}(t)
]
\]

### 6.3 模型选择

优先推荐：

- Gaussian Process：小数据、自然给出不确定度；
- 深度集成小型 MLP：数据量较大时使用；
- 随机森林集成：实现简单但梯度和时序连续性较弱。

基础论文可以不把残差预测作为核心贡献，而只用其不确定度决定更新幅值：

\[
\rho_k(t)
=
\frac{\rho_{\max}}
{1+\beta\sigma_k(t)}
\]

高不确定度区域允许更小修正。

## 7. 模块四：约束 QP 更新

### 7.1 实际误差与模型梯度分离

必须使用虚拟物理机床测得的误差：

\[
e_k^W=x_k^W-x_{\mathrm{ref}}^W
\]

而不是用名义模型预测误差代替。名义模型只提供 \(M_k^W\)。

### 7.2 推荐 QP

\[
\begin{aligned}
\min_{\Delta\kappa}\quad
&
\|e_k^W-M_k^W\Delta\kappa\|_{Q_W}^2\\
&+\eta\|e_k-M_k\Delta\kappa\|_{Q_G}^2\\
&+\lambda_\kappa\|\Delta\kappa\|_2^2
+\lambda_s\|D_2B\Delta\kappa\|_2^2
\\
\text{s.t.}\quad
&
u_{\min}\le u_k+B\Delta\kappa\le u_{\max}\\
&
v_{\min}\le D_1(u_k+B\Delta\kappa)\le v_{\max}\\
&
a_{\min}\le D_2(u_k+B\Delta\kappa)\le a_{\max}\\
&
j_{\min}\le D_3(u_k+B\Delta\kappa)\le j_{\max}\\
&
\|S\Delta\kappa\|_\infty\le\rho_k
\end{aligned}
\]

其中：

- 第一项是关键窗口任务误差；
- 第二项防止全轨迹明显恶化；
- 第三项限制单轮更新；
- 第四项抑制高频和不光滑修正；
- 最后一组约束是信赖域。

### 7.3 接受、回退和信赖域更新

定义预测改善：

\[
\Delta J_k^{\mathrm{pred}}
=J_k-\hat J_k(\kappa_k+\Delta\kappa_k)
\]

定义实际改善：

\[
\Delta J_k^{\mathrm{act}}
=J_k-J(\kappa_k+\Delta\kappa_k)
\]

比值：

\[
r_k=
\frac{\Delta J_k^{\mathrm{act}}}
{\Delta J_k^{\mathrm{pred}}+\epsilon}
\]

推荐逻辑：

- 若实际目标恶化超过容忍度，拒绝并回退；
- 若 \(r_k\) 很小或为负，缩小 \(\rho_k\)；
- 若 \(r_k\) 中等，保持 \(\rho_k\)；
- 若 \(r_k\) 高且更新触及边界，适当增大 \(\rho_k\)；
- 连续多轮 QP 不可行时，降低窗口数量或增加松弛变量。

阈值必须在验证集上确定并固定，不应针对每条测试轨迹手工调节。

## 8. 完整算法伪代码

~~~text
Input:
    reference path r(t)
    initial command u0(t)
    spline basis B
    nominal learner model f_hat
    motion constraints C
    maximum trials Kmax

Initialize:
    kappa_0 = 0
    trust-region radius rho_0
    history H = empty

for k = 0, 1, ..., Kmax - 1:
    1. Generate u_k(t) = u0(t) + B(t) kappa_k
    2. Execute u_k using LinuxCNC + virtual physical machine
    3. Log y_k(t), controller states and constraint states
    4. Compute task state x_k(t) and error e_k(t)
    5. Compute temporal features and criticality score s_k(t)
    6. Select and merge critical windows W_k
    7. Simulate nominal model and compute local Jacobian M_k
    8. Estimate residual uncertainty sigma_k(t)
    9. Set trust region rho_k from uncertainty and previous acceptance ratio
   10. Solve constrained QP for Delta kappa_k

    if QP is infeasible:
        relax optional constraints or shrink the active windows
        if still infeasible: stop with diagnostic

   11. Evaluate candidate command on the virtual physical machine
   12. Compare predicted and actual objective reduction

    if candidate is accepted:
        kappa_(k+1) = kappa_k - Delta kappa_k
        update residual model and trust region
    else:
        kappa_(k+1) = kappa_k
        shrink trust region and record rollback

   13. Stop if task tolerance, stagnation or trial budget criterion is met

Output:
    accepted final command
    full trial history
    critical windows
    convergence and safety diagnostics
~~~

## 9. 停止条件

满足任意条件时停止：

1. 关键窗口最大误差小于目标阈值并连续保持若干轮；
2. 连续 \(N_s\) 轮改善小于 \(\epsilon_J\)；
3. 达到最大试次数；
4. 连续 QP 不可行；
5. 连续回退次数超过上限；
6. 出现不可恢复的约束或数值异常。

最终输出应为“最佳已接受试次”，而不一定是最后一次候选试次。这对应原论文中继续学习可能导致性能退化的问题。

## 10. 方法实现顺序

### M0：无学习

验证 LinuxCNC、虚拟物理机床、日志和误差计算。

### M1：标准全轨迹 ILC

不做关键窗口，验证雅可比和 QP 更新链。

### M2：固定窗口 ILC

基于曲率或跃度阈值选窗口，建立可解释基线。

### M3：自动关键窗口 ILC

加入多特征评分、窗口选择和合并。

### M4：不确定度感知安全更新

加入残差估计、信赖域自适应和回退。

### M5：跨域初始化或迁移

在前述模块稳定后再增加，不应阻塞最小论文版本。

