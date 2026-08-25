# Dynamic Hypergraph TDCA v2.4.3：Graph-local EVC 与 Certified Meta-stop

日期：2026-08-25
状态：开发配置冻结；尚未运行 v2.4.3 provider 实验
训练：无
模型：Qwen-plus
随机种子：20260820

## 1. 研发动机

v2.4.2 已验证 horizon-aware causal credit 的目标定义具有正相关性，但 Smoke-A 在计划完成率、图证明完成率和 F1 上未通过冻结门槛。配对诊断显示两类失败：

1. choice-conditioned / min-max 资源成本被当作绝对成本，导致仍有充足预算的证明操作被裁剪；
2. operation-family delayed capacity 能改善总体相关性，却无法区分同一 family 内“通向终局证明”和“与终局断开”的操作；
3. `no_executable_computation` 缺乏可审计的死路证明，无法区分结构性不可行、资源耗尽和策略误停。

v2.4.3 不修改 v2.4.2 的 delayed credit target、因果归因规则和 `gamma=0.85`。本阶段仅替换成本语义、延迟价值的结构来源和 meta-stop 证据。

## 2. 绝对资源成本

对操作 (o) 和资源 (r)，定义：

[
c_r(o,t)=
operatorname{clip}_{[0,1]}
left(
rac{d_r(o)}{B_r}
left[
1+(m_{max}-1)
left(1-rac{R_r(t)}{B_r}ight)
ight]
ight),
]

其中 (d_r(o)) 是操作需求，(B_r) 是题级初始容量，(R_r(t)) 是选择时剩余容量，且 (m_{max}=2)。

总成本为：

[
C_{mathrm{abs}}(o,t)=
0.35c_{mathrm{call}}+
0.35c_{mathrm{token}}+
0.20c_{mathrm{retrieval}}+
0.10c_{mathrm{graph risk}}.
]

性质：

- 不使用 ready-set min-max normalization；
- 添加无关或 dominated operation 不改变已有操作成本；
- 资源越稀缺，成本单调不减；
- LLM call 成本不因 fidelity 降低而消失；
- token 成本随冻结的请求上限变化；
- predicted 和 actual cost 使用相同的容量与 scarcity 语义。

## 3. Controller-owned Proof Obligation

每个 obligation 独立保存：

- `obligation_id`；
- target subgoal 与 branch；
- obligation type；
- `OPEN / CLOSED / BLOCKED`；
- severity；
- terminal reachability；
- required/satisfied nodes；
- reason codes；
- provenance event IDs；
- creation/update step。

第一版 obligation 类型为：

- `missing_binding`；
- `missing_evidence`；
- `retrieval_exhausted`；
- `missing_claim`；
- `extraction_exhausted`；
- `missing_verification`；
- `missing_join_premise`；
- `terminal_disconnected_join`；
- `contradiction`。

状态只能由 `V2GraphController` 在成功 mutation 或 allocation reconciliation 后刷新。当前状态保存在 `proof_obligations`，每次刷新同时追加 `ProofObligationSnapshot`，以便审计关闭与阻塞过程。

每个 computation packet 显式记录 `target_obligation_ids`，因此调度决策必须说明其预计关闭的证明缺口。

## 4. Graph-local delayed value

v2.4.3 的 ranking 不再使用 operation-family delayed capacity。延迟价值来自当前图：

[
V_{mathrm{delayed}}(o)=
0.30O_{mathrm{closure}}+
0.25R_{mathrm{terminal}}+
0.20P_{mathrm{missing}}+
0.15R_{mathrm{candidate}}+
0.10R_{mathrm{evidence}}.
]

如果操作位于 terminal-disconnected 或 blocked 区域，则施加 dead-end risk 抑制。ANSWER commit 的 delayed value 保持为 0。

层级 family feedback 仍写入 trace 供诊断，但不进入 v2.4.3 的 EVC 或 budget packet。只有同一题、同一精确 region 的因果反馈允许影响后续计算。

## 5. Horizon-aware EVC

保持 v2.4.2 的 immediate/delayed 权重：

[
G(o)=0.40V_{mathrm{immediate}}(o)+0.60V_{mathrm{delayed}}(o),
]

[
operatorname{EVC}(o)=max(0,G(o)-C_{mathrm{abs}}(o)).
]

trace 同时保存 gross opportunity、各成本通道和 net EVC，禁止只保存最终标量。

## 6. Certified Meta-stop

决策顺序冻结为：

1. 检查是否已有满足独立 terminal belief hard gate 的 ANSWER；
2. 检查 operation 是否可执行和可负担；
3. 评估 gross proof opportunity；
4. 扣除绝对成本得到 net EVC；
5. 只有 net EVC 超过冻结阈值 0.08 才继续。

所有 ABSTAIN 或无可执行操作的终止必须附带 dead-end certificate，包含：

- OPEN/BLOCKED obligations；
- 每个候选操作预计关闭的 obligation；
- gross opportunity、absolute cost、net EVC；
- 剩余 calls/tokens/retrieval/graph operations；
- 未选择的可行 JOIN 数量；
- exhaustion/rejection evidence。

`ANSWER`、`ABSTAIN` 和 `BUDGET_EXHAUSTED` 继续保持互斥语义。

## 7. 兼容性

v2.4.3 字段采用显式 `proof_obligation_version`。旧图没有该版本时：

- 序列化不写入 v2.4.3 默认字段；
- seal payload 与冻结 artifact 保持一致；
- v2.4、v2.4.1、v2.4.2 图可继续校验和离线 replay。

## 8. 冻结离线结果

在 v2.4.2 Smoke-A 的 538 个 allocation candidates 上进行只读成本反事实：

- old choice-relative cost mean：0.15233；
- new absolute cost mean：0.06384；
- 恢复的成本裁剪候选：5；
- 两个预先登记的 chain-loss qid 均存在 net EVC > 0.08 的新成本反事实；
- 未读取 dataset、prediction、answer 或 gold metric。

该 replay 只验证成本语义。v2.4.2 trace 尚无 Proof Obligation，因此 graph-local delayed value 只通过结构测试验证，不能伪造历史反事实。

## 9. 准入顺序与停止条件

1. 全量零 API 测试；
2. 冻结 trace replay；
3. source-freeze commit 并推送；
4. v2.4.3 Smoke-A 20；
5. Smoke 全门通过才运行独立 Shadow-B 20；
6. 任一门失败，立即输出 safe-stop，禁止进入 v2.4.4；
7. Smoke 与 Shadow 均通过后，才实现 v2.4.4 signed hypergraph belief diffusion。

质量门保持：

- candidate presence ≥ 0.75；
- execution plan completion ≥ 0.75；
- graph proof completion ≥ 0.80；
- F1 ≥ 0.58；
- calls ≤ 163；
- tokens ≤ 185000；
- budget exhaustion rate ≤ 0.10；
- immediate correlation ≥ 0.10；
- delayed correlation ≥ 0.15。

新增硬门：

- proof-obligation trace completeness = 1；
- no-executable without certificate = 0；
- viable proof opportunity cost-clipping stop = 0；
- absolute-cost ready-set invariance = 1；
- 所有 ABSTAIN 均有 exhaustion evidence；
- leakage、invariant violation、controller mutation violation、unsupported ANSWER 均为 0。

组合 provider 预算不得超过 2000 attempts / 2,000,000 tokens。v2.4.2 已消耗 129 / 141651，因此 v2.4.3 子账本上限冻结为 1871 / 1858349。
