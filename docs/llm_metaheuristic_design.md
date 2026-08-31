# 基于大模型的元启发式算法自动设计 — 课题设计文档

> **课题**:用大语言模型(LLM,本地开源模型)在"组件设计空间"内进化元启发式算法,产出**超越手工设计的新元启发式算法**。
> **问题域**:混合流水车间调度(HFSP),优化目标 makespan。
> **模式**:框架 + 实证并重(方法贡献 + 对比实验)。
> 本课题暂停 RL 学习主线(见 §11),专注于 LLM 自动设计。

---

## 1. 课题概述

### 1.1 要解决什么问题

设计一个好的元启发式算法(GA/SA/IG/DPSO…)高度依赖**人类专家的领域知识**:选什么初始化、用什么算子、配什么接受准则、调什么参数。人手工搜索这个"设计空间"既慢又容易陷入思维定式。

本课题把**"设计算法"这件事交给 LLM**:LLM 作为进化搜索中的**变异算子**,在一个明确定义的**组件设计空间**里组合/改造候选算法,用真实算例上的效果(适应度)驱动进化,最终收敛出一个**人没想过、但效果更好**的新元启发式算法。

### 1.2 三个核心主张(论文的贡献点)

| # | 贡献 | 内容 |
|---|---|---|
| C1 | **组合式自动设计框架** | 把元启发式拆成"组件设计空间",LLM 只改结构化配置、不写原始代码 → 安全、可控、可复现 |
| C2 | **LLM 进化的实证优势** | LLM 进化出的算法 ≥ 手工 GA/IG/SA/DPSO 的对比结果 |
| C3 | **消融证明 LLM 与反馈的必要性** | LLM 变异 vs 随机变异、有反馈 vs 无反馈、有反思 vs 无反思 |

> C3 是审稿人最关心的:"为什么非用 LLM,随机搜索不行?" 消融实验必须回答。

### 1.3 为什么用"组件组合"而不是"LLM 写代码"

| 方案 | 优点 | 缺点 |
|---|---|---|
| LLM 生成完整算法代码 | 表达力最强 | 语法错误多、不可控、**不可复现**(同一 prompt 输出不稳),评测困难 |
| **组件组合(本课题)** | 合法空间小、执行确定、**可复现**、可直接用现有算子库 | 表达力受设计空间限制(但足以产生风格多样的算法) |

> "只改配置、不写代码" 本身是一个可写进论文的方法论贡献(相比 EoH/ReEvo 更工程可靠)。

---

## 2. 研究版图与定位

| 代表工作 | 方法 | 与本研究的关系 |
|---|---|---|
| FunSearch (DeepMind 2023) | LLM 搜索程序/函数,发现数学结构 | "生成→评测→反馈"范式的鼻祖 |
| EoH: Evolution of Heuristics (2024) | 进化 + LLM 当变异算子,生成启发式代码 | **本课题的主线模板**,但它是"生成代码",我们做"组合组件" |
| ReEvo (2024) | 在进化中加入反思(reflection),flow shop SOTA | 我们的反思模块参考 |
| LMEA (2023-24) | LLM 直接当进化算子(交叉/变异) | 备选对比 |
| AlphaEvolve (DeepMind 2025) | LLM+进化发现新矩阵算法 | 证明"LLM 造算法"范式成立 |

**差异点(本课题的新颖性)**:
1. 面向 **HFSP**(以上工作主要在 TSP/流水车间/装箱,HFSP 上系统做的少);
2. **组件组合式**设计空间(而非直接生成代码),执行可复现;
3. 基于**本地开源模型** + 显式的消融验证(LLM vs 随机、反馈 vs 无反馈)。

---

## 3. 核心思想:算法 = 组件组合

### 3.1 元启发式算法的分解

一个元启发式算法 = **结构开关 × 组件选择 × 参数**:

```
结构开关: 种群式(GA 味)/ 单解式(SA/ILS 味)/ 破坏-重建(IG 味)
组件:
  初始化:   NEH / SPT / LPT / 随机
  变异算子: swap / insert / inverse / scramble
  交叉算子: OX / PMX / 两点交叉(种群式时用)
  接受准则: 总是接受 / 只接受更优 / Metropolis(温度) / 阈值
  局部搜索: 开 / 关
参数:      population_size / max_iterations / destruction_size / temperature / 各概率
```

### 3.2 AlgorithmSpec(候选算法 = 一串配置)

LLM 操作的就是这个结构化对象(JSON):

```python
@dataclass
class AlgorithmSpec:
    name: str
    # ---- 结构开关 ----
    population_mode: bool            # True=种群(GA式) False=单解(SA/ILS式)
    use_destroy_repair: bool         # True=IG式破坏-重建
    destruction_size: Optional[int]  # use_destroy_repair 时有效
    # ---- 初始化 ----
    initializer: str                 # "neh" | "spt" | "lpt" | "random"
    # ---- 算子 ----
    mutation_operators: List[str]    # ⊆ {swap, insert, inverse, scramble}
    crossover_operators: List[str]   # ⊆ {ox, pmx, two_point}
    crossover_prob: float
    mutation_prob: float
    # ---- 接受准则 ----
    acceptance: str                  # "always" | "only_better" | "metropolis" | "threshold"
    temperature: Optional[float]     # acceptance="metropolis" 时有效
    # ---- 局部搜索 ----
    use_local_search: bool
    local_search_iterations: int
    # ---- 规模 ----
    population_size: int
    max_iterations: int
    elite_size: int
    tournament_size: int
```

不同的配置组合会真实长出**风格各异的算法**——GA 味、IG 味、SA 味,以及**人没想过的混合味**。这就是"新算法"的来源。

---

## 4. 组件池与执行器

### 4.1 组件池(直接复用现有代码)

| 组件 | 取值 | 现有实现 |
|---|---|---|
| 初始化 | neh / spt / lpt / random | `hfsp/methods/heuristics/` |
| 变异算子 | swap / insert / inverse / scramble | `hfsp/methods/operators/` 注册表 |
| 交叉算子 | ox / pmx / two_point | `hfsp/methods/operators/crossover.py` |
| 局部搜索 | 开 / 关 | `hfsp/methods/operators/local_search.py` |
| 搜索结构 | 种群 / 单解+破坏重建 / 单解 | `executor.py` 内置模板 |

> 设计空间 = 现有算子库 + 少量结构开关。**整个课题的"材料"就是你已有的代码**,不需要新造算法零件。

### 4.2 执行器 `ConfigurableMetaheuristic`(spec → 可运行的 Method)

把 `AlgorithmSpec` 解析成现有 `Method` 子类,实现统一的 `solve(instance) -> ScheduleSolution`:

```
if population_mode:
    → GA 式循环: 锦标赛选择 + 交叉 + 变异 + 精英保留
elif use_destroy_repair:
    → IG 式循环: 破坏 d 个 → NEH 式重插 → 接受准则 → 局部搜索
else:
    → SA/ILS 式循环: 变异 → 接受准则(metropolis/阈值/只接受更优)→ 局部搜索
```

**确定性要求**:spec + 固定 seed → 完全确定的执行与适应度(实验可复现的关键)。

> **设计空间的边界**:基于速度的 DPSO 无法用四个模板复现,故不作为种子 spec,而是留在对比实验里当外部基线。当前种子集 = GA / IG / SA-like / NEH+LS。

> **评估成本与时间预算**:元启发式(尤其 IG)的评估可能很慢。统一给每个 (spec, 算例, 运行) 设**相等的时间预算** `time_limit` 作为公平比较的标准做法(也保证运行时可控)。

---

## 5. 进化框架(EoH 路线)

```
输入: 训练算例集, 种群规模 N, 精英数 E, 代数 G, 重试上限 R
初始化种群:
  种子 = [GA, SA, IG, DPSO, NEH+局部搜索] 的 AlgorithmSpec
  补随机 spec 至 N
for gen = 1..G:
  # 1) 评估适应度(并行)
  fitness[spec] = 该 spec 在训练算例上的平均 RPD(固定 seed × K 次)
  # 2) 选择
  按 fitness 排序; 精英前 E 个直接进入下一代
  # 3) LLM 变异(补齐 N−E 个后代)
  for i in (N−E):
      parent = 锦标赛/加权选择的父代(单亲变异 或 双亲合并)
      feedback = {fitness, 平均 makespan, 收敛摘要, 反思点评(可选)}
      offspring = llm_mutate(parent, feedback, components_pool)
      校验: 非法 spec → 重试(最多 R 次)→ 仍失败则退回父代副本
  # 4) 反思(可选, ReEvo 风格)
  对优劣个体生成点评,作为下代变异上下文
输出: 历史最优 spec(在独立测试集上做最终评估)
```

### 5.1 LLM 变异提示(示意)

```
系统: 你是元启发式算法设计专家, 请在给定"组件池"内设计一个改进版算法配置。
用户:
  当前候选算法(JSON): {parent_spec}
  它在 HFSP 上的表现: 平均 RPD = {x}%, 平均 makespan = {y}
  反思: {reflection}   # 可选
  组件池(只允许这些取值): {components_pool}
  请输出改进版 AlgorithmSpec(JSON)。
```

### 5.2 反馈与反思的信息设计

| 反馈通道 | 内容 | 作用 |
|---|---|---|
| 适应度 | 平均 RPD / makespan / 收敛曲线摘要 | 让 LLM 知道"这个配置行不行" |
| 反思(可选) | 对淘汰个体的"为什么差"点评 | 引导下一轮变异方向(ReEvo 核心) |

---

## 6. LLM 调用层(本地开源模型)

### 6.1 后端抽象

设计统一接口 `LLMClient`,支持多种本地后端,**课题默认用 Ollama**:

```python
class LLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str,
             temperature: float = 0.8,
             json_mode: bool = True) -> dict:
        """返回解析后的 JSON 对象(AlgorithmSpec)。"""

class OllamaClient(LLMClient):      # 默认: ollama.chat, json 模式
class VllmClient(LLMClient):        # 备选: OpenAI 兼容接口
class TransformersClient(LLMClient):# 备选: HuggingFace pipeline
```

### 6.2 模型选择建议

- 任务性质:生成/修改 **结构化 JSON 配置**,对推理能力要求不高 → 不需要大模型。
- 建议: **Qwen2.5-7B / 14B** 或 **Llama-3.1-8B**(Ollama 一行拉取),中文友好、指令遵循好、够用。
- 结构化输出:Ollama 的 `format="json"` 强制 JSON + 本地 schema 校验 + 失败重试。

### 6.3 健壮性设计

- **校验**:spec → schema 校验,非法字段/非法取值 → 重试(≤R 次)→ 退回父代。
- **成本**:本地跑无 API 费用;单次调用毫秒~秒级,几百~几千次调用可承受。
- **可复现**:固定 temperature + seed(尽力而为);**执行层确定**才是复现的关键。

### 6.4 环境准备

```bash
# 安装 Ollama 并拉取模型
brew install ollama
ollama pull qwen2.5:7b        # 或 qwen2.5:14b / llama3.1:8b
# 启动服务
ollama serve
```

---

## 7. 评估与实验设计

### 7.1 数据隔离设计(分层拆分)

**Fernández-Viagas 480 基准按 (作业数, 阶段数) 分层,拆成训练集 + 测试集(黑盒)。进化和微调只用训练部分,测试部分从未见过。**

```
训练部分(336) ─► 进化 fitness ─┐
             ─► 微调数据     ─┼─► 最终算法 ─► 测试部分(144, 黑盒) ─► RPD vs UpperBounds
                               └─► 基线 IG / DABC / CSA
```

**拆分规则(已落盘 `benchmarks/seville/split/`)**:
- 分层 = (作业数, 阶段数),共 48 层,每层 10 个算例。
- 每层:idx 1~7 → 训练(336),idx 8~10 → 测试(144)。70/30,确定性、可复现。
- **进化训练子集(方案 B)**:从训练部分里按**尺寸 10~160 × 全部 4 个阶段 × 每格 1 个**抽样(`evolution_train_subset.txt`,40 个)→ 覆盖小到大尺寸。**200~240 作业只进测试(纯泛化)**:训练含 240 会让进化评估慢到 16-20 小时(实测),且时间预算机制已保证大算例在测试时获得足够时间,故 200-240 不需进训练。
- **K 折交叉验证**(可选):`kfold_split` 5 折,全部 480 都被测过一次(回答"全基准覆盖")。
- **K 折交叉验证**(可选):`kfold_split` 5 折,全部 480 都被测过一次(回答"全基准覆盖")。

**两条铁律**:
1. **进化和微调只能用训练部分** —— fitness 用训练子集算;微调数据(父/子 spec + 前后 fitness)也只来自训练部分。测试部分的算例**从未进入进化/微调**。
2. **最终在测试部分(144)上测的是"进化出的最终算法"**(不是 LLM 本身)—— 对比对象是 IG / DABC / CSA 等强基线。

**划分**:

| 用途 | 数据 | 说明 |
|---|---|---|
| 训练部分 | 480 里 336 个(idx 1-7/层) | 进化 fitness + 微调数据来源 |
| 进化训练子集 | 从训练部分分层抽 20 个(10~80 作业) | 评估快、覆盖尺寸分布 |
| 测试部分(黑盒) | 480 里 144 个(idx 8-10/层) | 最终评估,跨全部尺寸(含 120~240) |
| 参考值 | `UpperBounds_01_April_2019.xlsx` | 公开逐算例 best-known,RPD 分母,**也是进化 fitness 的绝对目标** |

**为什么**:同基准分层拆分 → 训练与测试分布一致(解决"20~50 作业训练 vs 40~240 测试"的失配);用已发表 UpperBounds 当 fitness 参考 → 种子停在 +5~10%,进化有真实的压向 0 的信号(此前用"种子最好"当参考,最好的种子天然 0%,进化无提升空间)。测试部分从未被进化/微调见过,仍是干净黑盒。

### 7.2 适应度定义

`fitness(spec) = 该 spec 在训练算例上、固定 seed 下、K 次运行的平均 makespan 的 RPD`:
`RPD = (mean_makespan − best_ref) / best_ref × 100%`
其中 `best_ref` 取 `UpperBounds_01_April_2019.xlsx` 中该算例的 Cmax(论文发布的上界/best-known)。统一时间预算 `t = ρ·n·m` 秒(ρ 常见 10/20/30),保证所有算法公平比较。

### 7.3 主实验

| 内容 | 说明 |
|---|---|
| 强基线 | speed-up IG (F-V 2022)、DABC (Pan 2014)、Chaos-SA (Lin 2021) |
| 弱基线 | GA、PSO/DPSO、NEH+局部搜索(框架已有) |
| 最终算法 | LLM 进化出的最优 spec,在 480 测试子集上跑 |
| 统计检验 | Friedman(多算法)+ Wilcoxon signed-rank(两两,含 Holm 校正) |
| 指标 | RPD 表、平均排名、胜率、最优值命中数 |

### 7.4 消融实验(C3)

| 消融 | 对比 | 回答的问题 |
|---|---|---|
| LLM vs 随机变异 | 同框架,变异源换成随机采样 | LLM 的"智能"变异是否必要 |
| 有反馈 vs 无反馈 | 不给 LLM 效果反馈 | 反馈回路是否必要 |
| 有反思 vs 无反思 | 关闭反思模块 | 反思是否带来增益 |
| 进化 vs 单点生成 | 一次性 LLM 生成 vs 迭代进化 | 进化是否值得 |

### 7.5 复现性

- 全局 seed 固定(`RNGManager`)。
- spec → executor 执行确定。
- 记录每代的完整轨迹(spec 历史 + fitness + 反思)供论文附录。
- **变异日志**(必做):每轮 LLM 变异记录 `(父spec, 子spec, 前后fitness, 算例集)` —— 既是可复现凭证,也是 M8 微调的数据源。

### 7.6 微调加分层(M8,可选但强烈建议)

**动机**:整条谱系(EoH/ReEvo/FunSearch + 现有 HFSP 工作)**无人微调过大模型**。用进化产生的真实奖励微调基座模型,是这条线的空白,可能成为论文最硬的创新点。

**数据来源(自产,不蒸馏强模型)**:
- 标签 = 真实 makespan(自己的评估器)—— 唯一 ground truth,不可替代。
- 样本 = 自己进化的变异日志 `(父spec → 子spec, 前后fitness)`,**只来自训练集,不碰 480**。
- 信号用"改进幅度/排名",不用绝对 fitness;数据覆盖 10~240 各档算例。

**微调方法**:
- LoRA / QLoRA 微调 qwen2.5:14b(租 GPU,单卡 24G+ 可跑)。
- SFT(只在"改进"对上)或 DPO(好坏偏好对)。

**验证(论文最有说服力的消融)**:微调后模型 vs 零样本模型,同一进化框架,比收敛速度与最终 RPD。

**风险**:奖励信号依赖算例(用改进幅度缓解);过拟合训练算例(数据覆盖各档 + 泛化验证);工作量与 GPU 预算。

**定位**:主线(零样本 P1~P5)保毕业;M8 做成则论文升级,做不成不影响主线。

---

## 8. 目录结构

```
hfsp/llm/
├── __init__.py
├── config.py                 # LLM 后端配置(模型名 / Ollama URL / temperature)
├── client.py                 # LLMClient 抽象: Ollama / vLLM / Transformers
├── prompts.py                # 提示词模板(系统 / 变异 / 反思)
├── representation.py         # AlgorithmSpec + JSON 校验
├── components.py             # 组件池注册与解析(spec 字段 ↔ 现有算子/启发式)
├── executor.py               # ConfigurableMetaheuristic: spec → 可运行 Method
├── evaluate.py               # 适应度评估(复用 ExperimentRunner + RPD)
├── evolve.py                 # 进化主循环(种群 / 选择 / LLM 变异 / 反思)
└── run.py                    # 全流程入口(读配置 → 进化 → 输出结果)

scripts/llm_evolve.py         # 命令行入口(可选)
docs/llm_metaheuristic_design.md   # 本文档
```

**依赖新增**:`ollama`(Python 客户端)或直接 HTTP 调 Ollama 的 REST API。其余全部复用现有框架。

---

## 9. 实施里程碑

| 阶段 | 内容 | 验证标准 |
|---|---|---|
| **P0** | 精读 EoH / ReEvo / FunSearch,写综述笔记 | 能讲清本研究与它们的三点差异 |
| **P1** | 组件池 + `ConfigurableMetaheuristic` executor | 5 个种子 spec 都能跑,结果合理且与 GA/IG/SA/DPSO 原版一致 |
| **P2** | 进化框架(先用**随机变异**打通闭环) | 搜索能收敛,最好 spec 优于随机基线 |
| **P3** | 接入 LLM(Ollama 本地模型)变异 + 反馈 | 小规模上 LLM 变体能改进父代 |
| **P4** | 反思模块 + 完整跑 G 代 | 产出最终算法,训练集上优于手工基线 |
| **P5** | 主实验 + 消融 + 论文写作(480 测试 + IG/DABC/CSA 对比) | RPD 对比表、消融表、统计检验 |
| **M8** | 微调基座模型(自产数据 + LoRA,可选加分层) | 微调 vs 零样本的消融;不碰 480 |

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| LLM 输出非法 spec | schema 校验 + 重试(≤R 次)+ 退回父代 |
| 本地模型能力不足(改不出好的) | 换更大模型(qwen 14B/32B);结构化任务对模型要求本就不高 |
| 评估太慢(每 spec 跑 12 算例 × K 次) | 进化期用小子集 + 短迭代上限 + 多进程并行 |
| 进化不收敛 | 缩小设计空间、加反思、调 N/G/K;先在小算例(10 作业)验证 |
| LLM 随机性影响复现 | 执行层确定性已保证;记录完整轨迹;温度调低做复现实验 |

---

## 11. 与 RL 学习的关系(本课题的取舍)

- 本课题**暂停 RL 学习主线**(`docs/rl_framework.md` 保留,后续可继续)。
- 但二者不冲突,反而互补:
  - 本课题的"评估 → 反馈 → 迭代"与 RL 的"奖励 → 更新"同源;
  - 将来可把 **RL 策略(DQN/PPO)作为组件或选择器嵌进本设计空间**,成为下一篇:"LLM + RL 联合自动设计算法"。
- 本课题优先,因为它是**离论文成果更近**的主线。

---

## 12. 一句话总结

> 把"设计元启发式算法"从**人**交给**LLM**:在组件设计空间内,LLM 作为进化变异算子,用真实算例效果驱动迭代,最终**自动产出一个超越手工设计的新元启发式算法**,并用消融实验证明"LLM + 反馈 + 反思"每一步都不可或缺。
