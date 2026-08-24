# Dynamic Hypergraph TDCA v2.3：smoke-20 安全停止报告

日期：2026-08-25
数据：MuSiQue distractor，冻结 smoke-20，seed `20260820`
模型：Qwen-plus，temperature `0.0`
结论：`NO_GO_FIX_BEFORE_CONTROLS`

## 1. 实验边界

本轮只在 development smoke-20 上开发和诊断。所有 gold-aware 信息仅用于运行后的失败归因，未进入检索、图状态、EVC、JOIN、终止或答案选择。v2.3.3 未通过预先声明的 smoke gate，因此没有运行 uniform/fixed-order 控制组，也没有扩大到 50/200/1000 样本。

当前 campaign 累计使用 648 次真实 provider attempts、722,979 个 provider-reported tokens，pending reservation 为 0；上限分别为 6,000 和 6,000,000。

## 2. 版本与结果

| 版本 | EM | F1 | Candidate | Full chain | Spearman(EVC, utility) | LLM calls | Tokens | Retrieval | Budget exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 稳定 v2.2 | 0.300 | 0.355 | 0.500 | 0.650 | -0.0553 | 159 | 179,035 | 63 | 0.100 |
| v2.3.2 | 0.400 | 0.484 | 0.700 | 0.600 | 0.2109 | 160 | 177,655 | 60 | 0.200 |
| v2.3.3 | 0.400 | 0.434 | 0.700 | 0.550 | 0.2845 | 188 | 207,940 | 77 | 0.200 |

v2.3.2 相对 v2.2 达到 candidate presence `+0.20`、F1 `+0.129`，且 EVC–utility 相关由负转正，但 full-chain 少 1/20。v2.3.3 试图同时修复 terminal 依赖缺口、空抽取恢复和 JOIN 前沿，结果未修复 full-chain，且 LLM calls 相对 v2.2 增长 18.2%，超过预声明的 10% 上限。

## 3. v2.3.3 的机制收益

v2.3.3 并非无效版本。与 v2.3.2 相比：

- 真实 operation choice rate：`0.1536 -> 0.2997`；
- cross-family choice rate：`0.1036 -> 0.2630`；
- Spearman(EVC, actual utility)：`0.2109 -> 0.2845`；
- JOIN attempts：`51 -> 32`；
- JOIN acceptance：`39.2% -> 62.5%`；
- JOIN model calls：`23 -> 1`；
- JOIN mean actual utility：`-0.0301 -> +0.0300`。

这些结果支持“图状态能够形成真实的跨区域、跨操作算力选择”这一机制判断，但 smoke 样本过小，不能形成论文效果声明。

## 4. 未通过 gate 的原因

### 4.1 结构闭包提高了安全性，但放大了低价值后续计算

`terminal_dependency_closure` 阻止 terminal root 绕过 planner 已生成但未完成的桥接子目标。这是正确的安全约束；此前存在“答案正确但中间子目标 pending”的结构性漏检。

然而闭包后更多子目标进入检索、抽取和验证流程：retrieval 从 60 增至 77，verification 从 54 墷至 76。第二次同子目标检索的平均 utility 为 `-0.0502`，最终答案证据使用率为 0，no-final-claim-yield 为 74.1%。新增计算主要流向了已饱和区域。

### 4.2 JOIN 生成质量改善，但 deterministic no-op 仍进入 allocator

32 次 JOIN 中有 12 次 `operation_produced_no_commit`；它们没有调用模型，却仍成为被选择的 operation，消耗 policy step 并延迟恢复动作。下一版必须在 ready-set 形成前执行与 `propose()` 一致的确定性 premise/support 检查，而不是选择后才失败。

### 4.3 空抽取恢复过宽

focused recovery 能避免一次 coverage 抽取为空后立即停止，但 v2.3.3 共记录 81 次抽取，空输出率 20.99%，accepted-attempt rate 66.67%。在证据没有变化时重复 coverage/direct-answer 仍可能连续为空，随后 editor 也可能 no-op。

### 4.4 当前 `full_chain_complete` 指标不是图证明完整度

现有通用评测中的 `full_chain_complete` 定义为 legacy execution plan 的全部 slot 是否 complete。它会受 coarse planner 输出的 slot 数和结构波动影响，并不等价于：

1. terminal claim 是否覆盖所有声明依赖；
2. JOIN proof closure 是否连通；
3. 每个 proof leaf 是否有独立 evidence；
4. answer 是否由该 closure 支撑。

本轮 gate 按预声明指标原样判负，不能在看到结果后更换指标。下一轮运行前应同时预注册 `execution_plan_completion` 与 `graph_proof_completion`，前者保留作兼容指标，后者作为论文中的推理能力主指标。

## 5. 下一轮的预注册修复顺序

在任何新 API 运行之前，依次完成：

1. **JOIN pre-allocation feasibility filter**：复用 premise status、absolute support、grounding、entailment、type 与 projection 条件，禁止必然 zero-call rejection 的 JOIN 进入 ready set。
2. **Region-level retrieval stopping**：首次检索后若已有 grounded claims，且同区历史 posterior/no-final-claim-yield 表明饱和，则第二轮检索必须与 extraction、verification、JOIN 或 abstain 同场比较，并施加可审计 saturation penalty。
3. **Bounded extraction recovery**：同一 evidence fingerprint 最多一次 coverage 和一次 direct-answer；两次均为空后不得重复抽取，只有新 evidence 或 graph revision 才能重开。
4. **Proof metric split**：实现并测试 `execution_plan_completion`、`graph_proof_completion`、dependency coverage 和 evidence-leaf coverage；在新 run 前冻结阈值。
5. **重新定义 smoke gate，但不得追溯修改本轮结论**：保留 zero leakage、zero invariant violation、zero unsupported answer、candidate `>= v2.2 +0.10`、F1 non-regression、positive EVC calibration、bounded calls/graph growth；同时分别报告 plan completion 与 graph proof completion。

完成上述零 API 修复和全量测试后，只允许再运行一次 adaptive smoke-20。只有该版本通过预注册 gate，才开放 matched uniform/fixed-order controls。

## 6. 可复核产物

- v2.3.2 run：`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787591050299957601`
- v2.3.3 run：`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787593956846465968`
- smoke 比较：`analysis_outputs/dynamic_v23_smoke_comparison/comparison.md`
- v2.3.3 离线诊断：`analysis_outputs/dynamic_v23_campaign/v233_offline/offline_diagnostic.md`
- paired 诊断：`analysis_outputs/dynamic_v23_smoke_comparison/paired_v233/paired_diagnostic.md`

所有三个分析脚本均声明并实际产生 `inference_calls_made = 0`。
