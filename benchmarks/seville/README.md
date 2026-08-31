# Fernández-Viagas & Framinan (2020) HFSP Testbed — 本地副本

来源:塞维利亚大学工业管理课题组官方页面
<https://grupo.us.es/oindustrial/en/research/results/>

论文: *Design of a testbed for hybrid flow shop scheduling with identical machines*,
Fernández-Viagas & Framinan, *Computers & Industrial Engineering* 141 (2020) 106288.

## 内容

| 路径 | 内容 | 规模 |
|---|---|---|
| `instances/small/` | 小规模算例 `instancia_<n>_<s>_<idx>.txt` | 240 个, n∈{10,15,20,25,30,35}, s∈{5,10,15,20} |
| `instances/big/` | 大规模算例 `instancia_<n>_<s>_<idx>.txt` | 240 个, n∈{40,80,120,160,200,240}, s∈{5,10,15,20} |
| `references/UpperBounds_01_April_2019.xlsx` | 逐算例上界(=论文发布时 best-known) | 480 行(2 sheet:Big_Size + Small_Size) |

## 算例文件格式

```
<n作业数> <s阶段数>
<m_1 ... m_s>                 # 每阶段同等并行机数量
<p_1_1 ... p_1_n>             # 阶段 1 行, n 个加工时间(作业维)
<p_2_1 ... p_2_n>             # 阶段 2 行
...                           # 共 s 行
```

- 阶段内的并行机**完全相同**(identical machines),加工时间只依赖 (作业, 阶段)。
- 文件名 `instancia_<n>_<s>_<idx>` ↔ 上界表的 `(n, s, Instance=idx)`。

## 上界表(references/UpperBounds_01_April_2019.xlsx)

- 列:`n`(作业数)、`s`(阶段数)、`Instance`(复现序号 1..10)、`Cmax`(上界)。
- 两个 sheet:`Big_Size`(240)、`Small_Size`(240)。
- 用途:作为 RPD 的分母(`RPD = (Alg − Cmax)/Cmax × 100%`)。
- 注意:这是论文用**改进 IG** 在 2019-04 算出的上界;Lin et al. (2021, *ESWA*) 的 Chaos-SA 声称又更新了部分上界(数值在付费正文)。

## 关联参考(完整清单见 docs/p0_literature_review.md)

- SOTA 算法: speed-up IG (Fernández-Viagas 2022)、DABC (Pan 2014)、Chaos-SA (Lin 2021)。
- 实验协议: 统一时间预算 `t=ρ·n·m`, RPD vs 上界表, 5~20 次运行, Friedman + Wilcoxon。

## 清理说明

本目录只保留**论文正式 480 算例 + 上界表**。页面上的 `testbed_1/2_*`、`best_known_*`、`upper_bounds_*`、`Results_*`、`CompleteEnumeration` 等属于**别的 testbed 或别的论文**,命名对不上这套 `instancia_*`,已删除;需要时可从官网重新下载。
