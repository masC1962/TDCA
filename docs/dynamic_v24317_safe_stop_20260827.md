# TDCA v2.4.3.17 安全停止记录（2026-08-27）

## 结论

本轮在全局硬上限前安全停止。v2.4.3.14 通过独立 Smoke-A，但未通过单臂
Shadow-B out-of-split replication；v2.4.3.17 在不回退 Smoke 指标的前提下，
修复了一类“由泛化陈述生成实体特定 claim”的 false-positive grounding 问题，
但尚未恢复被错误候选压制的正确答案。因此最终 TDCA gate 尚未关闭，Shadow-B
已经打开并用于诊断，下一轮必须构造全新且不重叠的 Shadow-C。

## 全局预算

- provider attempts：`1701 / 2000`
- provider-reported tokens：`1,990,701 / 2,000,000`
- 剩余：`299` attempts、`9,299` tokens
- 停止原因：剩余 token 不足以完成一次可审计的“实现—source freeze—完整 canary—gate”闭环。

## v2.4.3.14 独立 Smoke-A

- 20/20 完整，174 calls，205,051 tokens。
- F1 `0.4893`，candidate presence `0.65`，full-chain `0.60`，graph-proof `0.90`。
- 相对冻结 v1：candidate presence `+0.10`，full-chain `+0.10`。
- 9 个可审计 3/4-hop JOIN case，3 个下游实际使用 n-ary JOIN。
- 0 leakage、0 infrastructure/invariant/controller violation、0 unsupported answer。
- EVC trace、actual cost、non-uniform allocation 均完整。

## v2.4.3.14 Shadow-B

- 20/20 完整，168 calls，202,266 tokens。
- candidate presence `0.70`，说明候选召回可外推。
- full-chain `0.55`、graph-proof `0.85`、F1 `0.37`，低于 preregistered Smoke
  参考 `0.60 / 0.90 / 0.4893`，因此 gate 失败。
- 8 个可审计长链 JOIN，5 个下游 n-ary JOIN；安全、EVC 与终止类型仍全部合格。
- Shadow-B 在查看失败标签后成为诊断集，不能再用于无偏的下一版最终评估。

## v2.4.3.15 与 v2.4.3.16 的否决

- v2.4.3.15 使用完整 endpoint 字符串锚定，Smoke F1 从 `0.4893` 降至
  `0.3643`、full-chain 从 `0.60` 降至 `0.40`，否决。
- v2.4.3.16 允许非泛化实体 token（如 `Hastings`）锚定，答案 F1 恢复，
  但错误答案改为 ABSTAIN 导致 legacy full-chain 从 `0.60` 降至 `0.55`；按冻结
  non-regression gate 仍否决。该现象同时说明 legacy full-chain 会奖励错误链，论文实验
  应将 graph-proof completion 与 answer correctness 分开报告。

## v2.4.3.17

最终保留的规则只处理 universal/generic source span（`all/any/each/every/`
`generally/typically/usually`）：若其生成实体特定 dependent tuple，则 tuple 的
subject 与 value 必须在自身引用证据中有独立锚点。该规则不使用题号、答案或 gold，
不改变阈值，也不压缩 absolute support、relative weight、entropy 与 evidence gap。

- 冻结 Smoke cache replay：0 provider calls，所有主指标与 v2.4.3.14 完全相同。
- 非重叠 dev canary-5：F1 `0.60`、full-chain `0.80`、graph-proof `0.80`、
  candidate presence `0.60`，0 infrastructure failure，0 unsupported answer。
- 已打开 Shadow 单题因果诊断：错误 `municipality` support 从 `0.76` 降至
  `0.5025`，verified/grounded claim precision 从 `0.50/0.50` 升至 `1.00/1.00`；
  但正确 `Minas Gerais` support 为 `0.6810`，低于冻结 terminal `0.70`，最终安全
  ABSTAIN 而非正确 ANSWER。

## 下一轮停止目标

下一轮不应继续调 terminal threshold，也不应重用 Shadow-B。应实现独立的
query-conditioned semantic alignment channel，用于区分“证据支持一个真实 tuple”和
“该 tuple 完整回答当前 subgoal”两个事件，并保持 independent raw scoring：

1. 对每个 candidate 分别记录 evidence entailment 与 full-subgoal constraint coverage；
2. 将 qualifier、dependency binding 和 relation target 的覆盖缺口保留为独立 channel；
3. 用 dev-only counterfactual 验证正确候选 support 上升且错误候选不回升；
4. 冻结后在全新 Shadow-C20 上要求安全、结构与 F1 同时非回退；
5. 只有 Shadow-C 通过后，才进入 paired matched-compute controls、budget curve 和 Pareto gate。

停止条件是：新 Shadow-C 通过，或新的全局 provider hard cap 被安全触发；在此之前不得
宣称最终 TDCA gate 完成。
