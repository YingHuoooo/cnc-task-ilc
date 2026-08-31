# V11 五项新增实验包

本目录针对工作区中的 `V11增加实验.md`，按优先级 1–5 完成一套独立、可恢复、可审计的纯仿真实验。旧的 V8–V13 冻结结果和 `cnc_v11_paper_package` 均未修改。

## 五项工作与产物

1. **Matched ablation**：`results/01_matched_ablation/`
   - 1,800 次方法运行；
   - V11 Full、无残余时延对齐、Task-top2、Raw-top2、Uniform 五组；
   - 原始 JSONL checkpoint、CSV 和消融汇总。
2. **Domain-aware / hierarchical statistics**：`results/02_hierarchical_statistics/`
   - paired、domain-level、hierarchical bootstrap 并列；
   - 20,000 次固定种子重采样；
   - leave-one-domain-out 稳健性范围。
3. **Parameter sensitivity**：`results/03_parameter_sensitivity/`
   - 780 次方法运行；
   - γ、平滑窗口、样条控制点数、学习率的一维敏感性。
4. **Representative replay**：`results/04_representative_replay/`
   - 按最接近中位效应的预设规则自动选择案例；
   - 保存完整逐点命令、反馈、轮廓误差、时延和活动区历史。
5. **Virtual plant family**：`results/05_virtual_plant_family/`
   - 24 个 held-out LHS 对象和 6 个 challenge 对象；
   - 1,800 次方法运行；
   - plant 参数、逐运行结果、plant-level 效应和汇总。

## 推荐阅读顺序

1. `experiment_design.md`：输入、方法、规模、统计单位和冻结边界；
2. `protocol_pre_execution.json`：正式运行前写入的机器可读协议与源码哈希；
3. `analysis_report.md`：五项结果的中文分析；
4. `figure_contracts.md`：五幅图的结论、证据链和审稿风险；
5. `figures/`：每幅图的 SVG、PDF、600 dpi TIFF 和 300 dpi PNG；
6. `qa/validation_report.json`：完整性和一致性验收；
7. `MANIFEST.json`：文件大小和 SHA-256 清单。

## 复现命令

从工作区根目录执行：

```bash
env PYTHONPATH=cnc_task_ilc/src:v11_additional_experiments/scripts \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 V11_WORKERS=4 \
  /opt/anaconda3/bin/python3 v11_additional_experiments/scripts/run_experiments.py

env PYTHONPATH=cnc_task_ilc/src:v11_additional_experiments/scripts \
  /opt/anaconda3/bin/python3 v11_additional_experiments/scripts/analyze.py

/opt/anaconda3/bin/python3 v11_additional_experiments/scripts/validate.py
```

数值网格采用 JSONL 逐任务 checkpoint；再次运行会跳过已有 `job_id`。绘图后端固定为 Python/Matplotlib，不混用 R。

## 范围边界

- 仅为二维数值虚拟 CNC 机床模型；
- 未安装、配置或运行 LinuxCNC；
- 未运行 G 代码控制器链；
- 未连接真实机床、伺服或传感器；
- 未进行空运行、切削、尺寸或表面质量测量；
- LHS 参数范围是预定义数值不确定性族，不是由真实机床群体标定的概率分布。

