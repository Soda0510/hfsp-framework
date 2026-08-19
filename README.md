# HFSP Framework

混合流水车间调度问题（Hybrid Flow Shop Scheduling Problem, **HFSP**）研究框架。

一个模块化、可扩展的 HFSP 求解框架，覆盖从实例建模、精确/启发式/元启发式求解、批量实验管理到结果可视化的完整流程。

## 功能特性

- **实例建模**：从 Excel 读取 HFSP 实例（含基础加工时间、速度、能耗、交期、Q 表等扩展字段）
- **精确求解**：MILP（混合整数线性规划）
- **构造启发式**：NEH、SPT、LPT
- **元启发式**：GA、SA、IG、DPSO、Q-Learning
- **多目标优化**：NSGA-II、MOEA/D（Pareto 非支配排序 + 拥挤度距离 + Tchebycheff 分解）
- **目标函数**：Makespan、Flowtime、Tardiness、Energy、加权和聚合
- **批量实验**：多算法 × 多算例 × 多次独立运行，自动汇总（均值/标准差、RPD）
- **可视化**：甘特图、Pareto 前沿

## 安装

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

依赖：`numpy`、`pandas`、`matplotlib`、`openpyxl`、`pyyaml`、`scipy`、`tqdm`（Python ≥ 3.9）。

## 快速开始

### 单个算例 × 单个算法

```bash
# python scripts/run_single.py <算例名> <算法> [选项]
python scripts/run_single.py 10-5-6 NEH
python scripts/run_single.py 20-5-4 GA --time-limit 60 --seed 42
python scripts/run_single.py 10-3-3 IG --no-plot --output results/gantt.png
```

支持的算法：`NEH`、`SPT`、`LPT`、`GA`、`SA`、`IG`。

### Python 接口

```python
from hfsp.io import InstanceReader
from hfsp.methods.metaheuristics import GeneticAlgorithm

instance = InstanceReader("Data").load("10-5-6")
ga = GeneticAlgorithm(max_generations=200)
solution = ga.solve(instance)
print(f"Makespan: {solution.makespan:.1f}")
```

### 批量实验

```bash
# 指定算例与算法
python scripts/run_experiment.py --instances "10-*" --algorithms NEH,GA,IG --runs 5

# 全部算例
python scripts/run_experiment.py --all --runs 10 --time-limit 60
```

结果输出到 `results/`：`results.csv`（原始记录）、`summary.csv`（均值/标准差）、`rpd.csv`（相对百分比偏差）。

## 算例命名约定

算例文件存放在 `Data/` 目录，命名格式为 `<作业数>-<阶段数>-<每阶段机器数>`：

- `10-5-6`：10 个作业、5 个阶段、每阶段 6 台机器
- `20-3-4`：20 个作业、3 个阶段、每阶段 4 台机器

## 目录结构

```
hfsp/
├── core/              # 核心数据结构（HFSPInstance、Solution、Decoder）
├── io/                # 实例读取（Excel）
├── solvers/           # 精确求解（MILP）
├── methods/
│   ├── heuristics/    # 构造启发式（NEH、SPT、LPT）
│   ├── metaheuristics/# 元启发式（GA、SA、IG、DPSO、Q-Learning、NSGA-II、MOEA/D）
│   └── operators/     # 邻域算子（swap、insert、inverse、scramble、crossover、local_search）
├── objectives/        # 目标函数
├── experiment/        # 批量实验管理与统计
├── visualization/     # 甘特图、Pareto 前沿
└── utils/             # 随机数、计时、校验
scripts/               # 命令行入口（run_single、run_experiment、plot_comparison）
Data/                  # 算例（Excel）
```

## 目标函数

| 目标 | 说明 |
| --- | --- |
| `MakespanObjective` | 最大完工时间 |
| `FlowtimeObjective` | 总流经时间 |
| `TardinessObjective` | 总延迟（基于交期） |
| `EnergyObjective` | 能耗（节能调度） |
| `WeightedSumObjective` | 加权和多目标聚合 |
