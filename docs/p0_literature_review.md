# P0 文献综述笔记 — LLM 自动设计元启发式算法(HFSP)

> 本文档是课题的地基:锁定基准、SOTA、基线集、实验协议、LLM 自动设计谱系与论文定位。
> 由三路并行调研(基准/SOTA、HFSP 强基线、LLM 自动设计谱系)汇总,带引用。

---

## 1. 基准与 SOTA(我们拿什么当靶子)

### 1.1 基准确认: Fernández-Viagas & Framinan (2020) 480 算例 ✅

| | 小规模(240) | 大规模(240) |
|---|---|---|
| 作业数 | 10, 15, 20, 25, 30, 35 | 40, 80, 120, 160, 200, 240 |
| 阶段数 | 5, 10, 15, 20 | 5, 10, 15, 20 |
| 组合 | 6×4×10 = 240 | 6×4×10 = 240 |

- **来源**:"Design of a testbed for hybrid flow shop scheduling with identical machines", *Computers & Industrial Engineering* 141 (2020) 106288, DOI 10.1016/j.cie.2020.106288。
- **文件格式**(已实测):`instancia_<n>_<s>_<idx>.txt`,首行 `n s`,次行每阶段机器数,之后按阶段行 × 作业列的加工时间。每阶段为**同等并行机**。
- **本地**:`benchmarks/seville/Small_Size_Instances/` + `Big_Size_Instances/`(已下载解压)。
- **算例生成**:从 57,600 个随机候选里按"经验难度/充分性/统计友好性"选出 240+240;加工时间分布含 U(1, 40·mᵢ);机器配置 M1/M2/M3。

### 1.2 ✅ 已发表 best-known 找到了,且直接匹配 480 算例(重要)

**`UpperBounds_01_April_2019.xlsx`**(官网托管,两个 sheet:Big_Size + Small_Size,共 480 条):

- 列 = `n, s, Instance, Cmax`;命名与 `instancia_*` **完全一一对应**(已本地核实 480↔480 全匹配)。
- 小集 Cmax 360–2391,大集 765–12744。
- 这是论文用**改进 IG** 计算的上界 = 论文发布时(2019)的逐算例 best-known。
- 本地: `benchmarks/seville/UpperBounds_01_April_2019.xlsx`

> **这就是 RPD 的基准参考**:直接拿它当分母,完全公开、可复现、可辩护。
> 后续论文(如 Lin 2021 CSA)可能又更新了部分上界,但那需要从付费正文誊抄;先用 2019 上界做基参考,若你的算法或基线超过它,超出的就是"新 best-known"。

### 1.3 SOTA 算法(该基准上)

| 算法 | 出处 | 地位 |
|---|---|---|
| **改进 IG**(设定上界者) | Fernández-Viagas & Framinan (2020, C&IE) | 论文内最佳,上界来源 |
| **Chaos-SA (CSA)** | Lin et al. (2021, *ESWA* 183:115422) | **当前 best-known 设定者**(声称 ARPD 0.153% vs IG 0.291%)⚠️ 数字待原文核对 |
| **speed-up IG** | Fernández-Viagas (2022, *ESWA* 115903) | 明确超越 DABC,同线最强 |
| **DABC** | Pan et al. (2014, *Omega* 45:42–56) | 2020 前长期"此前最佳" |

**注意**:所谓 bounded-search IG 是为**分布式置换流水车间(DPFSP)** 提出的,不是 HFSP;HFSP 的 IG 主线是"普通 IG + 加速"。之前的笔记里把它当作 HFSP 基线是**错误**,已纠正。

### 1.4 建议的对比基线集(可直接辩护)

```
强基线(必用): speed-up IG(F-V 2022) 或 自实现 IG+加速、DABC(Pan 2014)、Chaos-SA(Lin 2021)
弱基线(常见): PSO/DPSO、GA(框架里已有)
参考基准:      UpperBounds_01_April_2019.xlsx(RPD 分母)
可选测试集:    Liao j30c5e1-10(30-60 作业)、Carlier-Néron 77(10-15 作业,小规模 sanity)
```

### 1.5 标准实验协议(论文必备)

- **指标**:RPD = (Alg − Best)/Best × 100%,多算例平均得 **ARPD**;Best 取上界表。
- **时间预算**:统一 `t = ρ·n·m` 秒(ρ∈{10,20,30},Pan 2014 沿用 Ruiz–Stützle 约定),或固定秒数——**必须所有算法同预算**。
- **重复**:每算例 5~20 次,报均值/最好。
- **统计检验**:**Friedman**(多算法整体)+ **Wilcoxon signed-rank**(两两,常加 Holm 校正)——HFSP 论文标配。

---

## 2. LLM 自动设计算法谱系

### 2.1 五个关键工作

| 工作 | 机制 | 问题 | 关键结果 | 与我们的差异 |
|---|---|---|---|---|
| **FunSearch** (DeepMind 2023, Nature 2024) | LLM 进化**程序/函数**,自动求值器打分,island 多样性 | 帽子集、在线装箱 | 帽子集 8 维 512 元素(20 年最大进步) | 生成代码,需数百万次采样 |
| **EoH** (ICML 2024 Oral) | "想法+代码"双表示**共同进化** | 在线装箱、TSP、**置换流水车间 FSSP** | 超手工启发式,LLM 查询量≈FunSearch 的 0.1% | 生成代码;我们只组合组件 |
| **ReEvo** (NeurIPS 2024) | 反思即 "verbal gradients",短期/长期反思引导进化 | TSP、CVRP、BPP 等(原论文**不含流水车间**) | TSP 上给出 SOTA 启发式 | 生成代码+专有 LLM |
| **LMEA** (CEC 2024) | LLM 直接当**进化算子**(选/交/变),解级进化 | TSP(≤20 节点) | 与传统启发式竞争力相当 | 进化的是解,不是算法 |
| **AlphaEvolve** (DeepMind 2025) | 进化式编码 agent,**整库进化**,Gemini 双模型 | 矩阵乘法、Borg 调度、硬件 | 4×4 复矩阵 48 次标量乘(Strassen 后首次) | 工程规模大,需快速确定可验证的求值器 |

### 2.2 ⚠️ 重要诚实发现:HFSP 上已经有人做"LLM 设计启发式"(2025-2026)

| 已有工作 | 场景 | 做了什么 |
|---|---|---|
| **Multi-Island-ReEvo** (Lei et al. 2026, Tsinghua Sci. & Tech.) | **动态 HFSP** | 多岛反思式进化生成调度**派工规则**,300 实例超 GP/GEP |
| **LLM-MAPPO** (2025, Adv. Eng. Inform.) | HFSP + 非同等并行机 | LLM 增强状态/动作,多智能体 DRL,330 实例 |
| **LLM-MODPPO-Evo** (2025, IEEE) | HFSP + 峰值功耗 | LLM 设计算法 + 进化 + 多目标 DRL |
| **LLM4DRD** (2026) | 柔性装配流水车间 | LLM 自动设计派工规则 |
| **LLM-AEA** | HFSP + 总延迟 | LLM 当超参数控制器 |

**结论**:这个方向不是空白。但已有工作几乎都集中在**动态 HFSP / 派工规则 / 变体目标(能耗、延迟、非同等机)**。

---

## 3. 论文定位(初步)—— 我们的差异点

| 维度 | 已有工作 | 我们的设计 |
|---|---|---|
| 设计空间 | LLM **写新代码**(EoH/ReEvo) | **组件组合式 spec**(从算子库选,不写代码)→ 执行确定、可复现 |
| 问题场景 | 动态 HFSP / 派工规则 / 变体目标 | **静态 HFSP + makespan + 标准 480 基准** |
| 模型 | 专有 LLM(GPT/Gemini) | **本地开源 qwen2.5:14b** |
| 验证 | 大多无消融 | **显式消融**:LLM vs 随机、反馈 vs 无反馈、反思 vs 无反思 |
| 参考 | — | **UpperBounds 2019 表**(公开可复现) |

**一句话定位**:*在静态 HFSP + 标准 480 基准上,用组件组合式设计空间 + 本地开源模型做 LLM 进化搜索,并用消融实验证明 LLM 与反馈回路的必要性。*

> 风险提示:审稿人可能拿 **Multi-Island-ReEvo(300 实例)**、**LLM-MAPPO(330 实例)** 的结果来对比。对策:在论文中明确"我们的范围是静态 makespan + 组件组合式,与动态派工规则 / 多智能体 DRL 不直接可比",并尽量在 480 基准上给出自己的完整结果。

---

## 4. 待办与风险

- [ ] **回原文核对转述数字**:Lin 2021 CSA 的 ARPD(0.153% vs IG 0.291%)是二手转述,需回原文确认。
- [ ] **Lin 2021 逐算例 best-known**:在付费正文表格,如要当作参考需人工誊抄(可选,优先级低)。
- [ ] **种子清单**:在付费正文 Section 3,不获取也不影响我们(我们直接用已下载算例)。
- [ ] **benchmark reader**:写 `hfsp/io/seville_reader.py` 解析 instancia 格式(P5 前)。
- [ ] **P4 反思模块**(研究主线下一步)。

---

## 5. 关键引用

- Fernández-Viagas & Framinan (2020), *Design of a testbed for hybrid flow shop scheduling with identical machines*, C&IE 141:106288 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S036083522030022X)
- 官方数据页(算例 + 上界表) — [grupo.us.es/oindustrial/en/research/results](https://grupo.us.es/oindustrial/en/research/results/)
- Lin, Cheng, Pourhejazy, Ying & Lee (2021), *New benchmark algorithm for hybrid flowshop scheduling with identical machines*, ESWA 183:115422 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417421008411)
- Fernández-Viagas (2022), *A speed-up procedure for the hybrid flow shop scheduling problem*, ESWA — [core.ac.uk](https://core.ac.uk/download/490594179.pdf)
- Pan, Wang, Li & Duan (2014), *A novel discrete artificial bee colony algorithm for the hybrid flowshop scheduling problem with makespan minimisation*, Omega 45:42–56
- Liu et al. (2024), *Evolution of Heuristics (EoH)*, ICML Oral — [PMLR](https://proceedings.mlr.press/v235/liu24bs.html) · [GitHub](https://github.com/FeiLiu36/EOH)
- Ye et al. (2024), *ReEvo*, NeurIPS — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)
- Romera-Paredes et al. (2024), *FunSearch*, Nature 625 — [Nature](https://www.nature.com/articles/s41586-023-06924-6)
- Liu et al. (2024), *LMEA*, CEC — [arXiv](https://browse.arxiv.org/abs/2310.19046)
- Novikov et al. (2025), *AlphaEvolve*, DeepMind — [博客](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- Lei et al. (2026), *Multi-Island-ReEvo (dynamic HFSP)* — [sciopen](https://www.sciopen.com/article/10.26599/TST.2026.9010050)
