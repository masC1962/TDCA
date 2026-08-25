# TDCA v2.4.2 Smoke-A 安全停止报告

日期：2026-08-25

## 结论

本轮研发按预注册规则结束于 `SAFE_STOP`。v2.4.2 的双时间尺度 EVC、provenance delayed credit、controller seal、完整 trace 和资源约束均通过，但 Smoke-A20 的三个质量 hard gate 未通过：

- execution plan completion：`0.65 < 0.75`；
- graph proof completion：`0.75 < 0.80`；
- F1：`0.5643 < 0.58`。

因此没有打开 Shadow-B，也没有运行 uniform/fixed controls 或 development50。本报告不修改已冻结阈值，也不把未通过的结果描述为成功。

## 运行身份与完整性

- Source commit：`23c1387ab3befa69910c4e62ef623fec0d7bbd15`
- Source tree SHA-256：`e8360bdefdb64b958ad6b69db193a9c8389c882c79943efef07470069a2089ad`
- Run：`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787652976991954408`
- Model：`qwen-plus`
- Seed：`20260820`
- Artifact verification：通过
- Infrastructure failures：`0`
- Graph invariant violations：`0`
- Controller-only mutation violations：`0`
- Unsupported answers：`0`
- Gold/oracle inference：`0`
- `failures.jsonl`：空

## 与 v2.4.1 的配对结果

| 指标 | v2.4.1 | v2.4.2 | 变化 |
|---|---:|---:|---:|
| Exact Match | 0.55 | 0.55 | 0.00 |
| F1 | 0.5843 | 0.5643 | -0.0200 |
| Candidate presence | 0.75 | 0.75 | 0.00 |
| Execution plan completion | 0.75 | 0.65 | -0.10 |
| Graph proof completion | 0.80 | 0.75 | -0.05 |
| Answered rate | 0.75 | 0.65 | -0.10 |
| Selective accuracy | 0.7333 | 0.8462 | +0.1129 |
| Logical LLM calls | 148 | 135 | -13 |
| Logical tokens | 167,857 | 146,230 | -21,627 |
| Budget exhaustion | 0.05 | 0.00 | -0.05 |

v2.4.2 保持了 EM 和 candidate presence，并降低了计算量、提高了回答子集上的精度，但以覆盖率、完整链、图证明和 F1 为代价。按照预注册标准，这不是可接受的 Pareto improvement。

## 已通过的机制门槛

总计 26 项 hard checks 中，23 项通过。除三项质量门槛外，其余均通过，包括：

- immediate EVC correlation：`0.4828 >= 0.10`；
- delayed EVC correlation：`0.2426 >= 0.15`；
- choice-conditioned delayed correlation：`0.0839 > 0`，有效选择数 `42 >= 20`；
- proof-gap operations：12；成功恢复 2，失败恢复 10；
- 成功恢复平均 delayed return：`0.4406 > 0`；失败恢复：`0.0`；
- complete EVC trace：`1.0`；
- complete delayed credit trace：`1.0`；
- non-uniform allocation：`1.0`；
- selected infeasible JOIN、重复 extraction fingerprint 和 no-diff editor allocation 均为 `0`；
- ANSWER/ABSTAIN/BUDGET_EXHAUSTED 类型保持分离。

这说明 v2.4.1 的 horizon mismatch 已被实质修复：预测 delayed value 与 provenance-realized delayed return 正相关，成功 recovery 获得正信用，失败 recovery 不会因时间先后获得伪信用。

## 预算账本

- Campaign hard cap：`2000 provider attempts / 2,000,000 provider-reported tokens`
- 本轮 provider attempts：`129`
- 本轮 provider-reported tokens：`141,651`
- Logical calls/tokens：`135 / 146,230`
- Cache hits：`6`
- Pending reservations：`0`
- 剩余 campaign allowance：`1871 attempts / 1,858,349 tokens`

未继续使用剩余预算，因为质量 hard gate 已经触发安全停止。

## 配对失败诊断

相对 v2.4.1：

- 11 条完整链保持；
- 3 条在两版中都不完整；
- v2.4.2 新获得 2 条完整链；
- v2.4.2 丢失 4 条原有完整链。

四条丢失链分成两类。

### 1. Cost clipping 导致过早停止：2 条

两题在仍有大量预算且仍存在可执行 extraction 时，净 EVC 被裁剪为 0：

- `4hop1__51465_53706_795904_580996`
  - gross horizon value：`0.3901`
  - normalized cost：`0.4584`
  - predicted EVC：`0`
  - 剩余预算：13 calls / 13,108 tokens / 6 retrievals
- `4hop2__161602_474028_88460_21057`
  - 两个 extraction candidate 的 gross horizon value：`0.4126`
  - normalized cost：`0.4945`
  - predicted EVC：`0`
  - 剩余预算：9 calls / 8,690 tokens / 4 retrievals

这表明当前 `normalized_cost` 与 gross horizon value 的尺度不匹配。问题不是预算真的耗尽，而是 choice-conditioned/min-max cost 作为绝对机会成本直接相减后过强。

### 2. Policy divergence 后 ready set 耗尽：2 条

- `3hop1__801799_547811_41132`
- `4hop1__152146_5274_458768_33637`

两题均产生了 accepted JOIN，但 JOIN 没有进入最终答案支持闭包，最后以 `no_executable_computation` 停止；v2.4.1 中对应 JOIN 被答案使用。新策略按 operation-family capacity 提供 delayed prior，虽然改善总体相关性，却没有足够细粒度地区分同一 JOIN/extraction family 内哪个 frontier 真正通向 terminal proof。

与此同时，v2.4.2 也在两题中从失败变为完整链，说明机制确实改变了搜索轨迹，但当前改变不具备稳定的质量安全性。

## 保留与冻结的结论

下轮不应回滚以下已验证部分：

- append-only controller-owned credit ledger；
- provenance-only eligibility；
- `gamma = 0.85` 的因果距离衰减；
- immediate/delayed/cost 三通道 trace；
- proof-gap 与 feasibility opportunity 的 bounded gain scaling；
- 已通过的 EVC/credit calibration gate 与阈值。

## 下一轮建议：v2.4.3

下一轮应只修 policy semantics，不重写 credit target：

1. 将 coarse operation-family delayed capacity 替换为 graph-local causal reachability 特征，例如 terminal dependency distance、可闭合 proof frontier、可完成 JOIN 的缺失 premise 数、当前 operation 到 terminal subgoal 的可达路径。
2. 重新定义 normalized cost。成本仍独立记录，但用于 meta-stop 的成本必须具有稳定的绝对尺度，不能直接把 choice-conditioned min-max 值当作跨 ready set 可比较的绝对机会成本。
3. 将“是否值得继续”与“ready operations 如何排序”分层：meta-stop 应检查 gross proof opportunity、remaining budget 和 calibrated absolute cost；排序再使用净 EVC。
4. 对 accepted-but-unused JOIN 增加 terminal proof reachability 信号，不增加单题规则或 gold-aware 补丁。
5. 先在冻结 trace 上做 counterfactual replay；随后制定新的 v2.4.3 preregistration，只运行新的 development Smoke-A。通过后才能重新考虑 Shadow-B。

## 可复现诊断文件

- Gate：`analysis_outputs/dynamic_v242_campaign/smoke_a_gate.json`
- Horizon audit：由 `scripts/analyze_dynamic_v242_offline.py` 生成
- Paired safe-stop diagnostic：`analysis_outputs/dynamic_v242_campaign/safe_stop_diagnostic.json`
- Diagnostic implementation：`scripts/analyze_dynamic_v242_safe_stop.py`
- Frozen protocol：`configs/dynamic_v242_preregistration.json`

本轮最终状态：`SAFE_STOP`。
