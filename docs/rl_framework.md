# HFSP 优化算法学习框架 — 目标框架设计文档

> 本项目的目的是**以混合流水车间调度(HFSP)为载体,系统学习从元启发式到深度强化学习的完整算法谱系**。
> 本文档是目标框架的设计蓝图 + 学习笔记:既覆盖已有元启发式(GA/SA/IG/DPSO/NSGA-II…),也包含新建的深度强化学习模块(DRL)。
> 先定框架,再逐模块实现。

---

## 1. 项目背景与学习目标

现有 `hfsp/` 框架已覆盖:实例建模、构造启发式(NEH/SPT/LPT)、元启发式(GA/SA/IG/DPSO)、多目标(NSGA-II/MOEA/D)、表格 Q-Learning(自适应算子选择)。

本模块在 `hfsp/rl/` 下**从零构建一套完整的 DRL 求解框架**,并用它串起一条 DRL 学习主线。

**学习目标**(按顺序掌握):
1. MDP 建模:把一个调度问题抽象成 状态/动作/奖励/转移
2. 理论地基:动态规划(策略迭代/值迭代)+ 表格 Q-Learning / Sarsa(§6.0,配 GridWorld 玩具环境)
3. 值函数方法:表格 Q-Learning → DQN → Double/Dueling/PER 三件套改进
4. 策略梯度方法:REINFORCE → 引入 Critic(A2C)→ GAE + clip(PPO)
5. 连续动作分支:DDPG → TD3 → SAC(配连续控制玩具环境,§6.5)
6. 三种调度决策范式如何映射到同一套 RL 骨架
7. LLM 收尾:把 RL(PPO)与 Attention 接到大型语言模型上(见 §14)

**优化目标**:Makespan(最小化最大完工时间)。后续可扩展 Flow time / Energy / 交期。

**问题规模**:小到中(10~20 作业)通用,通过归一化特征 + 固定长度表示适配。

**算法全景**:本项目把"求解一个 HFSP"要学的算法家族完整铺开(见 §1.1),DRL 只是其中一支、是这条线的终点。

### 1.1 算法全景图:本项目覆盖的算法家族

| 家族 | 算法 | 现有代码 | 核心思想 | 掌握判定标准 |
|---|---|---|---|---|
| 精确求解 | MILP | `hfsp/solvers/milp.py` | 建模为整数规划,分支定界求最优 | 能把小算例写成数学模型,并理解"为什么大规模不可行" |
| 构造启发式 | NEH / SPT / LPT | `hfsp/methods/heuristics/` | 贪心规则,秒出初始解 | 不看资料能默写 NEH 流程,能说清"NEH 为什么通常优于 SPT" |
| 单目标元启发式 | GA / SA / IG / DPSO | `hfsp/methods/metaheuristics/` | 邻域搜索 + 接受准则 / 种群进化,平衡探索-利用 | 能画全流程框图,能解释每个算子、每个参数存在的理由 |
| 多目标元启发式 | NSGA-II / MOEA/D | `hfsp/methods/metaheuristics/` | 非支配排序 + 拥挤度 / 分解为标量子问题 | 能讲清"非支配""拥挤度""分解"三个概念,并对比两者差异 |
| 深度强化学习 | DQN / REINFORCE / A2C / PPO | `hfsp/rl/`(本框架,待建) | 与调度环境交互学习策略,函数逼近 + 探索 | 见 §6 与 §13.1 的掌握判定 |

> 学习顺序建议:精确(MILP 理解建模)→ 构造启发式(会写)→ 元启发式(会用)→ 多目标(懂概念)→ DRL(本框架主线)。DRL 不是替代元启发式,而是另一条"从数据/交互中学"的路线,两者互为对比基准。

### 1.2 DRL 完整基础课程地图(基础全包含)

本框架的 RL 模块覆盖**现代 RL 的完整基础线**,不只在调度上。按分类维度铺开:

| 分类维度 | 分支 | 框架覆盖 | 算法 |
|---|---|---|---|
| 学什么 | 值函数 | ✅ | 动态规划值迭代/策略迭代、Q-Learning、Sarsa、DQN、Double/Dueling/PER |
| | 策略 | ✅ | REINFORCE(+baseline) |
| | Actor-Critic | ✅ | A2C、PPO |
| 用谁的数据 | On-policy / Off-policy | ✅ 都有 | PPO、REINFORCE / DQN、DDPG、TD3、SAC |
| 动作空间 | 离散 | ✅ | DQN、PPO(调度主线) |
| | 连续 | ✅ | DDPG、TD3、SAC(玩具环境,§6.5) |
| 环境模型 | Model-free | ✅ | 全部 |
| | Model-based | ⛔ 扩展 | 见 §11(非基础) |
| 数据来源 | 在线 | ✅ | 全部 |
| | 离线 / 模仿学习 | ⛔ 扩展 | 见 §11(非基础) |

> 一句话:基础 = 动态规划 → 表格 → DQN 家族 → 策略梯度 → Actor-Critic → 连续动作。学完这条线,"RL 的地图"就在你心里了。Model-based / 离线 / 模仿 / 多智能体属于进阶专题,先留到 §11。

---

## 2. 整体架构

```
hfsp/rl/
├── envs/                        # 环境层:把 HFSP 建模成 Gym 风格 MDP
│   ├── base_env.py              # 公共接口: reset()/step()/render()/close()
│   ├── dispatch_env.py          # 【范式1】L2D:机器空闲时选作业
│   ├── operator_env.py          # 【范式2】深度算子选择(表格 Q-Learning 的深度化)
│   ├── permutation_env.py       # 【范式3】端到端逐步生成作业排列
│   └── toy/                     # 学习用玩具环境(理论地基 / 连续分支)
│       ├── gridworld.py         # 表格方法用:策略迭代/值迭代/Q-Learning/Sarsa
│       └── pendulum.py          # 连续控制用:DDPG/TD3/SAC(经典 Pendulum)
│
├── networks/                    # 神经网络层
│   ├── mlp.py                   # 通用 MLP(可作 Q 网络 / 价值 V / 策略 π / 连续确定性策略 μ)
│   ├── attention.py             # Attention encoder(作业特征序列 → 上下文向量)
│   └── actor_critic.py          # Actor-Critic 双头网络(PPO/A2C 用)
│
├── agents/                      # DRL 算法层
│   ├── base_agent.py            # 抽象基类: select_action/update/save/load
│   ├── tabular/                 # ★ 理论地基:无神经网络,查表
│   │   ├── dynamic_programming.py  # 策略迭代 / 值迭代(需要模型)
│   │   ├── q_learning.py           # 表格 Q-Learning
│   │   └── sarsa.py                # 表格 Sarsa
│   ├── dqn.py                   # DQN(+Double DQN、Dueling DQN、PER 开关)
│   ├── reinforce.py             # REINFORCE 策略梯度(+baseline)
│   ├── a2c.py                   # Advantage Actor-Critic
│   ├── ppo.py                   # PPO(GAE + clip 裁剪)
│   ├── ddpg.py                  # ★ 连续动作:确定性策略梯度
│   ├── td3.py                   # ★ 连续动作:TD3(DDPG 改进:双 Q + 延迟更新 + 目标平滑)
│   └── sac.py                   # ★ 连续动作:SAC(最大熵)
│
├── buffers/
│   ├── replay_buffer.py         # 普通经验回放(off-policy, 离散/连续通用)
│   └── per_buffer.py            # Prioritized Experience Replay(带重要性采样权重)
│
├── trainers/
│   ├── base_trainer.py          # 公共训练循环: 环境交互/日志/评估回调
│   ├── off_policy_trainer.py    # DQN/DDPG/TD3/SAC 系: 回放采样 + 目标网络更新
│   └── on_policy_trainer.py     # REINFORCE/A2C/PPO: rollout 收集 + GAE + 多轮更新
│
├── evaluator.py                 # 评估: 与 NEH/SPT/LPT/随机策略对比, 算 RPD
└── utils/
    ├── features.py              # 状态特征构建 + 归一化
    ├── gae.py                   # GAE(广义优势估计)
    └── logger.py                # 训练曲线 / 指标记录(CSV + 可选 TensorBoard)
```

**设计原则**:
- **一套接口,三种范式**:所有环境继承 `HFSPBaseEnv`,统一 `reset/step/render`;训练器、回放、评估器全复用。
- **算法可插拔**:环境只暴露 `observation/action/reward`,不关心用什么算法;智能体只关心 `(s, a, r, s')`,不关心调度细节。
- **off-policy 与 on-policy 分流**:DQN 系用回放缓冲;策略梯度系收集 rollout 后更新。

---

## 3. MDP 建模(HFSP 问题的 RL 抽象)

一个调度问题变成 RL 问题,需要定义四要素。三种范式共享**底层调度模拟**,区别在"决策发生在哪、动作是什么"。

### 3.1 公共要素

- **底层模拟**:作业 `j` 依次经过 `s` 个阶段,每阶段有 `m_s` 台并行机器。每个作业在某阶段可分配任一台该阶段机器,加工时间 `p_{j,m}` 因机器而异。
- **状态信息的原材料**:
  - 机器可用时间 `available_m`(每台机器最早空闲时刻)
  - 作业进度 `completion[j, s]`(作业 j 已完成到哪一阶段)
  - 阶段排队情况:每个阶段当前可加工(但尚未被调度)的作业集合
  - 加工时间矩阵、作业剩余工作
- **奖励**:默认稀疏奖励,episode 结束时 `r = -makespan`(除以基准做尺度归一,如除以 SPT 解)。可选稠密奖励:每步 `-(当前最大完工时间) 的变化`。

### 3.2 范式1: L2D 作业选择(经典调度 DRL 路线)

| 要素 | 定义 |
|---|---|
| **决策点** | 某阶段某台机器空闲,且有 ≥1 个作业已就绪可加工 |
| **状态** | 当前部分调度特征:机器可用时间、就绪作业的各阶段加工时间、作业进度等(Attention encoder 编码) |
| **动作** | 从就绪作业中选一个,指派给空闲机器 |
| **转移** | 选中的作业在该机器上加工,时间推进到下一个最早决策点 |
| **终止** | 所有作业走完所有阶段 |
| **奖励** | 稀疏:`-makespan`;可加稠密分量 |

> 学习价值:这是「事件驱动的调度 RL」标准写法,对应 Zhang et al. (2019) "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" 的思路。

### 3.3 范式2: 深度算子选择(最易入门的 DQN 载体)

| 要素 | 定义 |
|---|---|
| **决策点** | 每轮元启发式迭代开始时 |
| **状态** | 种群特征向量:多样性、改进率、停滞代数(现有 `compute_state` 的连续化 + 归一化) |
| **动作** | 选择交叉/变异算子(swap/insert/inverse/scramble/crossover) |
| **转移** | 用选中算子对种群变异,下一代种群产生 → 计算新特征 |
| **终止** | 达到迭代上限 |
| **奖励** | 本轮 best-makespan 的改善量(稠密、每步都有) |

> 学习价值:从现有表格 Q-Learning 升级到「神经网络逼近 Q(s,a)」,是理解 **函数逼近替代查表** 的最佳切入点。状态是低维向量,用 MLP 即可,不需要 Attention。

### 3.4 范式3: 端到端排列生成

| 要素 | 定义 |
|---|---|
| **决策点** | 每次向排列追加一个作业 |
| **状态** | 已选作业集合(掩码)+ 全部作业特征(Attention encoder) |
| **动作** | 从"尚未入选"的作业中选下一个 |
| **转移** | 追加到排列尾部;排列完成后交给 `ListSchedulingDecoder` 解码出完整调度 |
| **终止** | 排列长度 = n |
| **奖励** | 稀疏:`-makespan`(解码后计算) |

> 学习价值:这是「组合优化 + Attention」路线,对应 Bello et al. (2017, Pointer Network 解决 TSP) 与 Permutation flow shop 的深度学习方法。难点在于**从部分解生成 → 完整调度**的两级结构。

---

## 4. 环境层接口约定(仿 Gym)

所有环境实现同一接口,训练器只依赖它:

```python
class HFSPBaseEnv(ABC):
    def __init__(self, instance, seed=None): ...

    @abstractmethod
    def reset(self) -> np.ndarray:
        """返回初始观测(归一化后的特征向量/特征矩阵)。"""

    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """执行动作, 返回 (next_obs, reward, terminated, info)。
        info 里放调试信息(如当前 makespan、决策点编号)。"""

    def render(self): ...
    def close(self): ...
```

**约定**:
- 观测统一为 `np.ndarray`,在 `utils/features.py` 里归一化到合理量级(如 [0,1] 或 z-score),保证不同规模算例特征尺度一致。
- 离散动作空间:动作编号 0..K-1,环境内部维护"当前合法动作掩码",网络输出用掩码屏蔽非法动作(范式1/3 每步合法动作数不同,必须掩码)。
- `seed` 贯穿环境内部 rng,保证可复现。

---

## 5. 神经网络层

### 5.1 `mlp.py` — 通用 MLP
- 输入:一维特征向量(范式2)
- 输出:可切换 —— `q_values`(DQN 用)/ `state_value`(critic 用)/ `action_logits`(policy 用)
- 结构:`Linear → ReLU → ... → Linear`(隐藏层数/宽度可配)

### 5.2 `attention.py` — Attention Encoder
- 输入:作业特征序列(N 个作业 × 特征维),外加当前部分调度上下文
- 结构:多头自注意力(self-attention)堆叠 → 每作业的编码向量 → 可再接一个 context 向量(query)做注意力打分
- 用途:
  - 范式1:对所有就绪作业打分,softmax 得选作业概率
  - 范式3:对未选作业打分,配合掩码
- 参考:Vaswani et al. (2017) Transformer 的 encoder 部分;可简化为单头/small 版本便于学习与训练。

### 5.3 `actor_critic.py` — Actor-Critic 双头网络
- 共享 backbone(MLP 或 Attention),两个 head:
  - `actor`:输出动作概率分布(离散 → softmax)
  - `critic`:输出标量状态价值 `V(s)`
- 用于 A2C / PPO。

---

## 6. DRL 算法层(学习主线)

### 6.0 理论地基:动态规划 + 表格方法(`agents/tabular/`,配 `toy/gridworld.py`)

所有深度 RL 的**数学根基**都在这里。没有神经网络,直接查表/迭代。强烈建议先跑通这部分再碰 DQN。

**贝尔曼方程(Bellman Equation)** —— 一切价值方法的源头:
`V(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [r + γ V(s')]`

- **策略迭代**(Policy Iteration):交替做「策略评估(算当前策略的 V)→ 策略改进(贪心更新 π)」,直到收敛。
- **值迭代**(Value Iteration):把上式变成 `V(s) ← max_a Σ P [r + γ V(s')]`,直接迭代到不动点。
- **Q-Learning(off-policy,表格)**:`Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`。学的是"不管现在用什么策略,目标是最优策略"。
- **Sarsa(on-policy,表格)**:把上式里 `max_a'` 换成"用当前策略实际选的 a'"。学的是"当前策略的价值"。

> 学习价值:DQN 的三个技巧(回放/目标网络/ε-greedy)都是在"表格 Q-Learning 直接搬到神经网络"出了问题之后加上的补丁。**先看懂表格版,才能看懂 DQN 为什么要打这些补丁。**

### 6.1 DQN 家族(`dqn.py`)

**理论起点**:表格 Q-Learning 的更新是 `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]`。
当状态连续/复杂,无法查表 → 用神经网络 `Q_θ(s,a)` 逼近。

**DQN 三个关键技巧**:
1. **经验回放**:打破样本相关性,提高样本利用率,稳定训练。
2. **目标网络**:用滞后网络 `Q_θ⁻` 计算 TD 目标,减小自举带来的震荡。每 C 步同步 `θ⁻ ← θ`。
3. **ε-greedy**:以 ε 概率随机探索,逐渐退火。

**三个改进(开关控制)**:
- **Double DQN**:用 `θ`(在线网)选最优动作、`θ⁻`(目标网)估价值,缓解高估:
  `Q_target = r + γ Q_θ⁻(s', argmax_a Q_θ(s', a))`
- **Dueling DQN**:把 Q 拆成 价值+优势 `Q(s,a) = V(s) + A(s,a) − mean(A)`,学习更稳定。
- **PER(优先级回放)**:TD 误差大的样本采样概率更高,`P(i) ∝ |δ_i|^α`,并用 `w_i = (N·P(i))^−β` 修正偏差。

**伪代码(DQN 训练循环,off-policy)**:

```
初始化 Q_θ, Q_θ⁻ = Q_θ, 空回放 D
for episode in range(E):
    s = env.reset()
    while not done:
        a = ε-greedy(Q_θ(s))          # 掩码掉非法动作
        s', r, done, info = env.step(a)
        存 (s, a, r, s', done) 入 D    # done 要特殊处理: 终态 Q 目标为 r
        每隔 N 步, 从 D 采样 batch, 计算 TD 目标, 梯度更新 θ
        每 C 步同步 θ⁻ ← θ
```

### 6.2 REINFORCE(`reinforce.py`)

**核心公式(策略梯度定理)**:
`∇J(θ) = E[ Σ_t G_t · ∇log π_θ(a_t|s_t) ]`,其中 `G_t = Σ_{k} γ^k r_{t+k}` 是折扣回报。

**关键点**:
- 直接优化策略 `π_θ(a|s)`,不需要 Q 网络。
- 高方差 → 引入 **baseline b(s)** 降方差:`∇J = E[ Σ_t (G_t − b(s_t)) ∇log π_θ(a_t|s_t) ]`。baseline 可用状态价值的蒙特卡洛估计。
- 属于 **on-policy**,一条轨迹用完即弃(或重要性采样修正)。

### 6.3 A2C(`a2c.py`)

**思想**:用 Critic 网络 `V_φ(s)` 估计价值,把 REINFORCE 里的回报 `G_t` 换成 **优势** `A_t = G_t − V_φ(s_t)`。
优势"这个动作比平均水平好多少",比绝对回报更稳、方差更低。

- Actor 更新:`∇J = E[ A_t · ∇log π_θ(a_t|s_t) ]`
- Critic 更新:回归到回报 `L(φ) = (V_φ(s_t) − G_t)²`
- **A2C**(A-synchronous 改为同步):多个并行环境收集样本 → 求平均梯度 → 统一更新,训练更稳定、GPU 利用率高。

### 6.4 PPO(`ppo.py`)

**解决 A2C 的一个问题**:步长不好设,大了训练崩、小了太慢。

**clip 目标函数**(近端策略优化核心):
```
L(θ) = E[ min( ρ_t A_t,  clip(ρ_t, 1−ε, 1+ε) A_t ) ]
ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   # 新旧策略比值
```
- 当 `A_t > 0`(好动作):鼓励,但 `ρ_t` 超过 `1+ε` 就截断 → 防止一步更新太大。
- 当 `A_t < 0`(坏动作):抑制,`ρ_t` 低于 `1−ε` 就截断。
- 配合 **GAE(广义优势估计)** 计算优势,权衡偏差/方差。

**GAE 公式**:
`A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`,其中 `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`
- `λ=0` → 只看一步 TD(偏差大方差小)
- `λ=1` → 蒙特卡洛回报(偏差小方差大)
- 常用 `λ=0.95` 折中。

**PPO 训练循环(on-policy)**:

```
for iteration:
    for env 并行 rollout: 收集 {s, a, logπ_old, r, s'}
    用 GAE 计算优势 A_t 与回报 G_t
    for K epochs:                    # 同一批数据多轮利用
        采样 minibatch, 计算 clip loss + value loss + entropy(鼓励探索)
        梯度更新 actor 与 critic
    丢弃旧轨迹, 重新收集            # 样本用完即弃(重要性修正保证近似)
```

### 6.5 连续动作分支:DDPG / TD3 / SAC(`agents/`,配 `toy/pendulum.py`)

调度是离散动作,但**连续动作算法是现代 DRL 的半壁江山**(机器人、自动驾驶、能源调度都有连续量)。这套独立在玩具环境上学习,不进调度主线。

**共同点**:off-policy + 经验回放 + 目标网络;都是 Actor(确定性策略 μ(s) 或随机策略 π(a|s)) + Critic(Q 函数)。

| 算法 | 核心思想 | 相对前者的改进 |
|---|---|---|
| **DDPG** | 连续版的 DQN:确定性策略 `a = μ_θ(s)`,Critic `Q(s,a)` 评估 | 把 DQN 扩展到连续动作(用 μ 代替 argmax) |
| **TD3** | DDPG 的三处修补 | ① 双 Critic 取小值(缓解 Q 高估)② 延迟更新 Actor ③ 目标动作加噪声(平滑) |
| **SAC** | 最大熵:目标 = 回报 + 熵 | 熵正则让策略更随机、更稳、样本效率更高;目前最流行的连续算法 |

**SAC 目标函数(最大熵 RL 核心)**:
`J(π) = Σ_t E[ r_t + α·H(π(·|s_t)) ]` —— 在最大化回报的同时,鼓励策略"随机一些"(α 是熵权重,可自动调节)。

**DDPG 的 Q 目标**:
`y = r + γ Q_θ⁻(s', μ_φ⁻(s'))` —— 用目标网络的 μ 产生下一个动作,再用目标网络 Q 估值。

---

## 7. 缓冲层

| 缓冲 | 用途 | 算法 |
|---|---|---|
| `replay_buffer.py` | 均匀采样经验回放(离散/连续通用) | DQN 基础版、DDPG、TD3、SAC |
| `per_buffer.py` | 按 TD 误差优先级采样 + 重要性权重 | Double/Dueling + PER |

---

## 8. 训练器设计

### 8.1 off-policy(DQN 系)
- 每个决策步与真实环境交互,存入回放
- 每 `learn_every` 步,从回放采样 batch 更新
- 每 `target_update` 步同步目标网络
- ε 随训练进度线性/指数退火

### 8.2 on-policy(REINFORCE / A2C / PPO)
- 收集一整条 rollout(或多条并行 rollout)
- 算 GAE → 多 epoch 更新 → 丢弃旧数据
- PPO 对同一批数据做 K 次内层更新(clip 限制步长)

---

## 9. 评估与实验

`evaluator.py` 负责统一评测:

- **基准对比**:随机策略、NEH、SPT、LPT、GA(元启发式)。算 RPD:
  `RPD = (f_algo − f_ref) / f_ref × 100%`,参考值取当前最好解(或 MILP 下界,小规模时)。
- **指标**:平均/最差/最好 makespan、训练曲线、每 episode 的求解时间。
- **输出**:`results/rl/` 下 CSV + 对比图(复用 `visualization/`)。
- **泛化验证**:在训练未见过的算例(不同作业数/阶段数)上测试,观察泛化能力(DRL 的重要课题)。

---

## 10. 实施里程碑

按学习难度递进,每一步可独立跑通、可验证:

| # | 里程碑 | 产出 | 学习重点 |
|---|---|---|---|
| M0 | 吃透已有元启发式 | 手写 NEH / GA 核心流程;读懂 SA / IG / DPSO / NSGA-II / MOEA/D | "探索-利用"框架,这是 DRL 的思想前身与对比基准 |
| M1 | 理论地基 | `tabular/`(策略迭代/值迭代/Q-Learning/Sarsa)+ `toy/gridworld.py` | 贝尔曼方程、表格方法,DQN 的一切补丁都在回应这里的痛点 |
| M2 | 公共骨架 | `base_env` + `replay_buffer` + `dqn` + `off_policy_trainer` + 随机策略跑通闭环 | 环境接口、训练循环结构 |
| M3 | 范式2 算子选择 DQN | `operator_env` + `mlp` + DQN 跑通并优于随机 | 表格 RL → DQN 的跨越 |
| M4 | 范式1 L2D 作业选择 | `dispatch_env` + `attention` + DQN | Attention 状态编码、动作掩码 |
| M5 | 算法层补全 | `reinforce` / `a2c` / `ppo` + `on_policy_trainer` + `gae` | 策略梯度全谱系 |
| M6 | 范式3 排列生成 | `permutation_env` + 对接解码器 | 两级结构(生成→解码) |
| M7 | 连续动作分支 | `ddpg` / `td3` / `sac` + `toy/pendulum.py` | 连续动作三件套(DDPG→TD3→SAC 的改进链) |
| M8 | 评估与实验 | `evaluator` + 基准对比 + 泛化测试 | RPD、实验设计、结果解读 |
| M9 | LLM 收尾 | LLM 生成启发式 / 调度 Agent / RLHF 实验(见 §14) | 把 RL(PPO)与 Attention 接到大模型 |

---

## 11. 后续扩展(暂不纳入主线)

基础线(动态规划 → 表格 → DQN 家族 → 策略梯度 → Actor-Critic → 连续动作)已全部纳入主线(见 §1.2)。以下是**进阶专题**,基础打牢后再碰:

- **多智能体 RL(MARL)**:多工厂 / 多目标交互场景。
- **Model-based RL**:用环境模型做想象 roll-out(MCTS、Dyna、MuZero 思想)。
- **离线 RL / 模仿学习**:只用固定数据集学(CQL、IQL)、从专家示范学(BC、GAIL)。
- **分层 RL(HRL)**:把大任务拆成高层决策 + 低层执行。

> LLM 相关已单列为主线终点,见 §14。

---

## 12. 文件清单(实现目标)

```
hfsp/rl/
├── __init__.py
├── envs/{__init__, base_env, dispatch_env, operator_env, permutation_env}.py
├── envs/toy/{__init__, gridworld, pendulum}.py
├── networks/{__init__, mlp, attention, actor_critic}.py
├── agents/{__init__, base_agent, dqn, reinforce, a2c, ppo, ddpg, td3, sac}.py
├── agents/tabular/{__init__, dynamic_programming, q_learning, sarsa}.py
├── buffers/{__init__, replay_buffer, per_buffer}.py
├── trainers/{__init__, base_trainer, off_policy_trainer, on_policy_trainer}.py
├── evaluator.py
├── utils/{__init__, features, gae, logger}.py
└── configs/                     # 每范式 × 每算法的默认超参 YAML(可选)
```

依赖新增:`torch`。其余复用现有 numpy / pandas / matplotlib。

---

## 13. 学习路线图(给读者)

```
第0步 M0 吃透现有元启发式 —— 对照 §1.1, 手写 NEH/GA, 读懂 SA/IG/DPSO/NSGA-II —— 这是 DRL 的对比基准和思想前身
第1步 M1 理论地基 —— GridWorld 上手跑通策略迭代/值迭代/Q-Learning/Sarsa, 吃透贝尔曼方程 —— DQN 的每个补丁都源于表格版的痛点
第2步 看懂 §3 的 MDP 抽象 —— 先理解"调度怎么变成 RL 问题"
第3步 M2 骨架 + 随机策略跑通 —— 建立"环境 ↔ 智能体 ↔ 训练器"的心智模型
第4步 M3 算子选择 DQN —— 亲手体验 表格→函数逼近 的跨越, 看懂 §6.1 三个技巧
第5步 M4 L2D —— 理解 Attention 如何编码"部分调度", 学会动作掩码
第6步 M5 策略梯度三兄弟 —— 顺着 REINFORCE→A2C→PPO 看每步在解决什么缺点
第7步 M6 排列生成 —— 把"生成排列 + 解码"两级结构吃透
第8步 M7 连续动作 —— Pendulum 上跑通 DDPG→TD3→SAC, 看懂确定性/随机策略在连续空间的差异
第9步 M8 实验对比 —— 学会用 RPD / 训练曲线 / 泛化实验 评价一个 DRL 调度方法
第10步 M9 LLM 收尾 —— 见 §14: 把 PPO / Attention 接到大模型, 完成从优化到 LLM 的闭环
```

每看完一章理论,回到对应代码读注释;每跑通一个里程碑,回到本表打勾。

### 13.1 掌握判定自测(对一个算法的最终检验)

对任意一个算法(NEH / GA / DQN / PPO …)的掌握,最终检验是——**挑一个你没见过的变体问题**(如:加机器随机故障、加交期权重、改流水线为柔性车间),然后:

1. 不查资料,独立设计/改造出一个能用的算法;
2. 能说清每一步设计**为什么**这么选;
3. 跑通之后,效果差时能**自己定位是哪个环节**(建模 / 算子 / 参数 / 编码)并改进。

能做到这三点 → 这个算法你真正掌握了。各档次的判定标准见对话中的"掌握阶梯"。

---

## 14. LLM 阶段:从优化算法到大型语言模型(项目终点)

学完 M0~M6(元启发式 + DRL)之后,本项目进入 LLM 收尾。目标不是"再学一个模型",而是把你已有的 **PPO / Attention / 优化** 三块能力接到大模型上,形成"调度 → RL → LLM"的完整闭环。

### 14.1 为什么这是自然延伸,不是换赛道

| 你已学的东西 | 在大模型中的对应物 |
|---|---|
| **PPO**(§6.4,手写实现) | RLHF 的训练算法就是 PPO。对齐大模型的机制和你训练调度策略是同一套 |
| **Attention**(§5.2,手写实现) | Transformer 的骨架。加 causal mask 即 decoder-only(GPT 结构) |
| **组合优化背景** | LLM for 组合优化(LLM-as-optimizer)是热点,纯 NLP 背景的人缺领域问题,你有 |

### 14.2 三个可选方向(建议按序做,递进)

**方向 A:LLM 生成 / 改进调度启发式**(最贴近现有代码,见效最快)
- 用 LLM 生成候选启发式规则 / 算子(如新的 NEH 变体),用现有 `hfsp` 环境批量验证,保留有效解(LLM-as-optimizer)。
- 产出:一条"LLM 生成 → 环境验证 → 择优"的自动搜索管线。

**方向 B:LLM Agent 调度员**(练 Agent / 工具调用 / RAG 工程)
- 让 LLM 通过工具调用(`run_single.py`、MILP 求解器、你的 DRL 策略)逐步求解一个 HFSP 实例,像 ChatGPT 用工具那样。
- 产出:一个能"读算例 → 选算法 → 求解 → 解读结果"的调度 Agent。

**方向 C:RLHF 对齐实验**(最硬核、最"大模型")
- 用你自己写的 `ppo.py`,在一个小语言模型(如 GPT-2)上做"监督微调 → 奖励模型 → PPO 对齐"的完整链路。
- 产出:亲手复现大模型对齐流程——这是转大模型岗位最硬的敲门砖。

### 14.3 推荐路径

先做 **A**(复用现有代码,快速验证"LLM + 优化"想法)→ 再做 **B**(补齐 Agent/RAG 工程)→ 有余力做 **C**(RLHF,把 PPO 用到语言模型上)。A + B 已足够撑起一篇毕业论文;C 是冲刺大模型对齐岗位的加分项。

### 14.4 与岗位能力的对应

| LLM 阶段产出 | 对应岗位能力 |
|---|---|
| A:LLM + 优化管线 | LLM 应用 / 评估方法论 |
| B:调度 Agent | Agent 工程 / 工具调用 / RAG |
| C:RLHF 实验 | 大模型对齐 / 后训练(最稀缺) |

### 14.5 阶段目标

跑完 M7,你就拥有了一条完整的个人能力链:**懂优化问题 → 会手写元启发式 → 会手写 DRL(PPO)→ 能把 LLM 接到具体问题**。这比"只会调 LLM API"或"只会刷题"都更有辨识度。
