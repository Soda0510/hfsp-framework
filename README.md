# HFSP Framework

混合流水车间调度问题（Hybrid Flow Shop Scheduling Problem, **HFSP**）研究框架。

一个模块化、可扩展的 HFSP 求解框架，覆盖从实例建模、精确/启发式/元启发式求解、批量实验管理到结果可视化的完整流程。适用于调度算法研究、算法对比与多目标/节能调度等方向的实验。

## 功能特性

- **实例建模**：从 Excel 读取 HFSP 实例（含基础加工时间、速度、能耗、交期、Q 表等扩展字段）
- **精确求解**：MILP（混合整数线性规划，基于 Gurobi）
- **构造启发式**：NEH、SPT、LPT
- **单目标元启发式**：GA、SA、IG、DPSO
- **多目标优化**：NSGA-II、MOEA/D（Pareto 非支配排序 + 拥挤度距离 + Tchebycheff 分解）
- **强化学习辅助**：Q-Learning 自适应算子选择
- **目标函数**：Makespan、Flowtime、Tardiness、Energy、加权和聚合
- **批量实验**：多算法 × 多算例 × 多次独立运行，自动汇总（均值/标准差、RPD）
- **可视化**：甘特图、Pareto 前沿、算法对比图

## 安装

```bash
pip install -r requirements.txt
# 或可编辑安装
pip install -e .
```

依赖：`numpy`、`pandas`、`matplotlib`、`openpyxl`、`pyyaml`、`scipy`、`tqdm`（Python ≥ 3.9）。

> **注意**：MILP 精确求解需要额外安装 [Gurobi](https://www.gurobi.com/) 及 `gurobipy`（`pip install gurobipy`）。其余算法不依赖它。

## 快速开始

### 1. 单个算例 × 单个算法（命令行）

```bash
# python scripts/run_single.py <算例名> <算法> [选项]
python scripts/run_single.py 10-5-6 NEH
python scripts/run_single.py 20-5-4 GA --time-limit 60 --seed 42
python scripts/run_single.py 10-3-3 IG --no-plot --output results/gantt.png
```

命令行支持的算法：`NEH`、`SPT`、`LPT`、`GA`、`SA`、`IG`（其余算法见下文 Python 接口）。

### 2. Python 接口

```python
from hfsp.io import InstanceReader
from hfsp.methods.metaheuristics import GeneticAlgorithm

instance = InstanceReader("Data").load("10-5-6")
ga = GeneticAlgorithm(max_generations=200)
solution = ga.solve(instance)
print(f"Makespan: {solution.makespan:.1f}")
print(f"Flow Time: {solution.flow_time:.1f}")
```

### 3. 批量实验（命令行）

```bash
# 指定算例与算法
python scripts/run_experiment.py --instances "10-*" --algorithms NEH,GA,IG --runs 5

# 全部算例
python scripts/run_experiment.py --all --runs 10 --time-limit 60

# 含多目标算法
python scripts/run_experiment.py --instances "10-5-6" --algorithms NEH,GA,DPSO,NSGA-II,MOEA/D --runs 5
```

结果输出到 `results/`：

| 文件 | 说明 |
| --- | --- |
| `results.csv` | 每次运行的原始记录（makespan、flow_time、tardiness、energy、runtime 等） |
| `summary.csv` | 按算例 × 算法汇总的均值/标准差 |
| `rpd.csv` | 相对百分比偏差（RPD，以 NEH 为基准） |

### 4. 算法对比图

```bash
python scripts/plot_comparison.py   # 读取 results/results.csv，生成 results/comparison_chart.png
```

## 算法

所有元启发式算法继承自 `Method`，统一接口 `solve(instance) -> ScheduleSolution`，并共享 `rng`（随机数生成器）与 `time_limit`（时间上限，秒）参数以支持可复现实验。

### 构造启发式

| 算法 | 函数 | 说明 |
| --- | --- | --- |
| NEH | `neh_heuristic` | Nawaz–Enscore–Ham 的 HFSP 变体：按总加工时间降序，逐个插入到使 makespan 最小的位置 |
| SPT | `spt_heuristic` | 最短加工时间优先：按总加工时间升序 |
| LPT | `lpt_heuristic` | 最长加工时间优先：按总加工时间降序 |

三者签名一致：`f(instance, decoder=None, rng=None) -> ScheduleSolution`。

### 单目标元启发式

#### GA — 遗传算法（`GeneticAlgorithm`，name=`"GA"`）

排列编码 + 锦标赛选择 + 精英保留，支持多种交叉/变异算子。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `population_size` | int | 80 | 种群规模 |
| `crossover_prob` | float | 0.8 | 交叉概率 (0..1) |
| `mutation_prob` | float | 0.3 | 变异概率 (0..1) |
| `max_generations` | int | 500 | 最大迭代代数 |
| `elite_size` | int | 2 | 精英保留个数 |
| `tournament_size` | int | 2 | 锦标赛选择规模 |
| `use_local_search` | bool | False | 是否对精英解做局部搜索 |
| `decoder` / `rng` / `time_limit` | — | — | 通用参数 |

#### SA — 模拟退火（`SimulatedAnnealing`，name=`"SA"`）

NEH 初始化 + 随机邻域扰动 + Metropolis 接受准则。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `initial_temperature` | float | 100.0 | 初始温度 |
| `cooling_rate` | float | 0.97 | 降温系数 (0..1) |
| `max_iterations` | int | 300 | 每个温度层的内循环次数 |
| `max_total_iterations` | int | 50000 | 总迭代次数上限 |

#### IG — 迭代贪婪（`IteratedGreedy`，name=`"IG"`）

基于 Ruiz & Stützle (2007) 的破坏-重建框架：NEH 初始化 → 随机移除 d 个作业 → NEH 式重插 → 局部搜索 → 温度接受。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `destruction_size` | int | `None` | 破坏规模 d，默认 `max(2, n_jobs//10)` |
| `temperature` | float | `None` | 接受温度，默认自适应（5% 劣解约 50% 接受率） |
| `max_iterations` | int | 2000 | 最大迭代次数 |
| `use_local_search` | bool | True | 重建后是否做局部搜索 |

#### DPSO — 离散粒子群（`DiscretePSO`，name=`"DPSO"`）

基于排列的离散 PSO，速度以「保持当前 / 趋向 pbest / 趋向 gbest」的概率形式作用于工件排序。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `swarm_size` | int | 50 | 粒子数 |
| `max_iterations` | int | 500 | 最大迭代次数 |
| `w` | float | 0.5 | 惯性权重（保持当前位置概率） |
| `c1` | float | 0.3 | 认知系数（趋向 pbest 概率） |
| `c2` | float | 0.2 | 社会系数（趋向 gbest 概率） |

### 多目标优化

多目标算法同时优化 **makespan** 与 **flow time** 两个目标，`solve()` 返回最接近理想点的「膝点」（knee point），完整 Pareto 前沿存放在 `algorithm.pareto_front`。

#### NSGA-II（`NSGAII`，name=`"NSGA-II"`）

Deb et al. (2002)：非支配排序 + 拥挤度距离 + 二元锦标赛 + 父子合并截断精英。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `population_size` | int | 100 | 种群规模 |
| `crossover_prob` | float | 0.9 | 交叉概率 |
| `mutation_prob` | float | 0.3 | 变异概率 |
| `max_generations` | int | 500 | 最大代数 |
| `tournament_size` | int | 2 | 锦标赛规模 |

#### MOEA/D（`MOEAD`，name=`"MOEA/D"`）

Zhang & Li (2007)：Tchebycheff 分解为 N 个标量子问题，邻域内竞争更新，外部种群（EP）归档 Pareto 前沿。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `population_size` | int | 100 | 子问题数（= 权重向量数） |
| `H` | int | 99 | Das & Dennis 权重向量划分数（2 目标时 pop_size ≈ H+1） |
| `T` | int | 20 | 邻域规模 |
| `delta` | float | 0.9 | 从邻域选择父代的概率 |
| `nr` | int | 2 | 一个子代最多替换的邻域解数 |
| `crossover_prob` / `mutation_prob` / `max_generations` | — | 0.9 / 0.3 / 500 | 同 NSGA-II |

### 强化学习辅助

#### Q-Learning（`QLearningAgent`）

用于元启发式的**自适应算子选择**：把种群状态（多样性、改进率、停滞代数）离散化为状态，用 ε-greedy 选择交叉/变异算子。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `n_operators` | int | 必填 | 候选算子数量 |
| `n_states` | int | 8 | 离散状态数 |
| `alpha` | float | 0.1 | 学习率 |
| `gamma` | float | 0.9 | 折扣因子 |
| `epsilon` | float | 0.2 | 初始探索率 |
| `epsilon_decay` | float | 0.995 | 每轮探索率衰减 |
| `initial_q_table` | ndarray | `None` | 预训练 Q 表 |

辅助函数：`compute_state(diversity, improvement_rate, stagnation, n_states)`、`compute_diversity(population)`。

### 精确求解

#### MILP（`MILPSolver`）

基于 `gurobipy` 的混合整数线性规划精确求解，适用于小规模算例（n ≤ 20）验证启发式解的下界。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `time_limit` | float | 300.0 | 求解时间上限（秒） |
| `mip_gap` | float | 0.0 | 相对 MIP 最优间隙 |
| `verbose` | bool | False | 是否打印求解日志 |

```python
from hfsp.solvers.milp import MILPSolver
solution = MILPSolver(time_limit=60).solve(instance)
```

## 目标函数

所有目标函数继承自 `ObjectiveFunction`，接口 `compute(solution) -> float`（均求最小化）。

| 类 | `name` | 公式 / 说明 | 依赖数据 |
| --- | --- | --- | --- |
| `MakespanObjective` | `makespan` | `C_max = max_j C_j` | — |
| `FlowtimeObjective` | `flow_time` | `F = Σ_j C_j` | — |
| `TardinessObjective` | `tardiness` | `T = Σ_j max(0, C_j − d_j)` | `due_dates` |
| `EnergyObjective` | `energy` | 加工能耗 + 空闲/关机能耗（含 break-even 关停决策） | `power_on`、`power_idle`、`power_reset`、`break_even_point` |
| `WeightedSumObjective` | `weighted_sum` | 归一化加权和 `Σ w_i · (f_i / ref_i)` | 可配置权重与参考值 |

`WeightedSumObjective` 构造参数：`w_makespan=1.0`、`w_flowtime=0.0`、`w_tardiness=0.0`、`ref_makespan=1.0`、`ref_flowtime=1.0`、`ref_tardiness=1.0`。

## 求解结果：`ScheduleSolution`

`Method.solve()` 返回的 `ScheduleSolution` 主要属性：

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `permutation` | list[int] | 工件加工顺序 |
| `assignments` | list[dict] | 每道工序的 `{job, stage, machine, start, end}` |
| `makespan` | float | 最大完工时间 |
| `flow_time` | float | 总流经时间 |
| `tardiness` | float | 总延迟（需交期数据） |
| `energy` | float | 总能耗（需能耗数据） |
| `rank` / `crowding_distance` | — | 多目标算法的 Pareto 元数据 |
| `job_completion_times` | property | 各作业在末阶段的完工时间 |
| `machine_schedules` | property | 按机器分组的调度 |

## 算例命名约定

算例文件存放在 `Data/` 目录，命名格式为 `<作业数>-<阶段数>-<每阶段机器数>`：

- `10-5-6`：10 个作业、5 个阶段、每阶段 6 台机器
- `20-3-4`：20 个作业、3 个阶段、每阶段 4 台机器

## 目录结构

```
hfsp/
├── core/              # 核心数据结构（HFSPInstance、ScheduleSolution、Decoder）
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
