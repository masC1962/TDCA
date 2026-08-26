# Dynamic Hypergraph TDCA v2.4.3.4：Smoke-A 结果与 v2.4.3.5 迭代边界

日期：2026-08-26

训练：无

模型：Qwen-plus

数据：MuSiQue distractor，冻结 Smoke-A20，seed = 20260820
v2.4.3.4 源代码冻结：`287b3f147acbc1c0d381581c6d82416284d58a21`

## 1. v2.4.3.4 机制与结果

v2.4.3.4 将已经通过独立 terminal belief readout 的 `commit:answer`
定义为零 provider 调用的确定性状态物化。证书不信任操作类型或 `accepted=true`，而是从
sealed graph 重算 absolute support、evidence gap、contradiction、type consistency、proof
coverage，并从完整候选竞争快照重算 relative weight、margin 与 entropy。

运行目录：
`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787731034484847607`

结果为 20/20 完整、0 failures、150 provider attempts、177,664 provider-reported tokens。
主要指标如下：

| 指标 | v2.4.3.3 | v2.4.3.4 | Gate |
|---|---:|---:|---:|
| Candidate presence | 0.60 | 0.70 | 0.75 |
| Execution-plan completion | 0.65 | 0.65 | 0.75 |
| Graph-proof completion | 0.90 | 0.85 | 0.80 |
| F1 | 0.00 | 0.475 | 0.58 |
| Answered rate | 0.00 | 0.55 | — |
| Selective accuracy | — | 0.818 | — |
| Immediate EVC Spearman | 0.543 | 0.580 | 0.10 |
| Delayed EVC Spearman | 0.091 | 0.141 | 0.15 |
| Choice-conditioned delayed Spearman | -0.186 | -0.431 | $>0$ |

11 个 terminal materialization 全部被执行，ANSWER / ABSTAIN 为 11 / 9，unsupported answer
仍为 0。这证明终局状态物化缺口已修复，并且没有绕过 conjunctive terminal gate。

## 2. 失败诊断

本轮存在三个一般性机制缺口：

1. allocator 对包含 3 个候选的 assignment BRANCH 签发 transition certificate，但 medium
   fidelity 在执行时只物化 2 个候选。两次证书因此承诺 3 个 child、实际得到 2 个 child；
   这是 certificate 与 concrete action 未绑定，而非图控制器执行失败。
2. child branch 的合法证明可以引用 parent branch 创建的 claim。v2.4.3.4 terminal certificate
   却要求所有 claim 的 `branch_id` 等于当前 child，导致两个已通过 terminal readout 的答案
   在状态物化前停止。正确约束应是 claim branch 位于 sealed parent lineage，而不是任意跨分支。
3. delayed predictor 只刻画静态 obligation tractability，没有使用当前题内 operation/region 的
   已观察结果。冻结 trace 上旧 predictor 的整体/真实选择子集 Spearman 为 0.141/-0.431。
   归一化 closure probability、expected delta、observed value、cooldown、redundancy 和 dead-end
   的确定性乘积为 0.197/+0.048，无训练、无 gold。

另外，6 个 proof-gap recovery 均使用含 “Independent source”“Gap” 和最多 5 个实体的
检索元指令。它们虽返回 1–3 个新 passage，却没有形成 candidate/JOIN/terminal 后继。查询应保留
推理目标，只使用最多两个图锚点和少量关系词，避免元指令稀释 BM25/dense 表达。

## 3. v2.4.3.5 冻结修改

下一轮只包含以下一般性修改：

1. 在 fidelity 截断后对 concrete operation 重新计算并记录 transition certificate；
2. terminal certificate 允许严格的 sealed branch-ancestry claim 继承，仍禁止 sibling/cross-branch；
3. delayed value 改为：

$$
V_{\mathrm{delayed}} =
P_{\mathrm{close}}\,\Delta O\,V_{\mathrm{observed}}
(1-C_{\mathrm{cooldown}})(1-R_{\mathrm{redundancy}})(1-R_{\mathrm{deadend}}).
$$

4. proof-gap retry 改用紧凑 objective query，最多两个 graph-state anchors，不含答案、gold 或
   “Independent source / Gap” 元指令。

Meta-stop threshold、terminal thresholds、JOIN gate、per-question safety cap 均保持不变；不使用
训练、单题规则或答案适配。v2.4.3.5 Smoke-A 不通过时继续结构诊断，不开启 Shadow-B。
