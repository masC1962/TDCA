# TDCA v2.4.2：双时间尺度 EVC 与因果信用协议

日期：2026-08-25
状态：开发配置冻结，尚未调用 v2.4.2 推理 API

## 1. 研究问题

v2.4.1 将 proof-gap recovery 引入动态超图调度后，在 Smoke-A 上提高了候选覆盖、完整推理链、图证明完成率和 F1，但预注册的总体
Spearman 相关性未通过。失败的直接原因不是恢复操作无效，而是预测量与监督量处于不同时间尺度：检索或抽取操作可能在当前 step
只产生低即时效用，却通过后续 VERIFY、JOIN 和 ANSWER 完成证明。

v2.4.2 不修改 gold-free 推理原则，也不训练价值模型。它将调度价值拆成三个可独立审计的通道：

$$
\hat U(o)=0.4\,\hat U_{\mathrm{immediate}}(o)
          +0.6\,\hat R_{\mathrm{delayed}}(o)
          -\hat C(o).
$$

三个通道均先归一化到 \([0,1]\)。`predicted_evc` 只用于排序；raw components、normalized components、预算请求以及三个独立通道全部保留。

## 2. 延迟信用的因果约束

延迟信用只沿显式 provenance 传播。合法因果边包括：

- evidence 到引用该 evidence 的 claim；
- dependency claim 到下游 claim；
- hyperedge premises 到 JOIN conclusion；
- supporting claims/evidence 到 answer；
- 节点 provenance 中记录的 source nodes 到新节点。

仅仅“较早发生”不会建立 eligibility。执行 DAG 中的 subgoal 顺序也不会自动成为 evidence credit 边，避免把同一题内所有后续成功错误归因给早期操作。

对 allocation \(a\) 的延迟回报为：

$$
R_a = \frac{\sum_k w_k r_{a,k}}{\sum_k w_k},
\qquad
r_{a,k}\leftarrow \gamma^{d(a,k)}r_{a,k},
\qquad \gamma=0.85.
$$

其中五个独立分量及开发权重为：

| 分量 | 权重 | 定义 |
|---|---:|---|
| proof completeness | 0.35 | 因果后继 accepted answer 的 chain coverage |
| candidate availability | 0.20 | 后续 proof-usable candidate 的 absolute support |
| accepted evidence | 0.20 | 后续非 proposed claim 实际使用的因果 evidence 比例 |
| successful JOIN | 0.15 | 后续有效 JOIN conclusion |
| supported terminal answer | 0.10 | 后续通过 terminal belief gate 的 accepted answer |

上述量不读取 gold answer。Gold 只在运行完成后的 EM/F1 评估器中使用。

## 3. Append-only credit ledger

每次 controller 完成 allocation reconciliation 后，为发生变化的旧 allocation 追加一条 `CreditAssignmentRecord`。旧记录不被覆盖。题目终止时，每个 allocation 必须有 terminal credit marker。

每条记录包含：source allocation、source operation、source/observation step、credit seeds、全部 causal descendants、逐节点因果距离、raw/normalized delayed components、realized delayed return、causal event IDs 和 attribution version。

信用账本属于 `controller_state_hash` 的密封状态。重复 ID、逆时间记录、缺失 allocation、非法距离、越界回报或 controller 外修改都会触发 graph invariant failure。

## 4. Fidelity 修复

v2.4.1 将 proof-gap reducibility 和 feasibility unlock 乘以 `gain / fraction`，使低 token fidelity 可能获得大于 1 的机会信号。v2.4.2 仅在新配置中改为乘以 `gain = sqrt(fraction)`，并强制限制在 \([0,1]\)。冻结的 v2.4.1 及更早配置保持原语义。

## 5. 预测 delayed horizon

终局 ANSWER COMMIT 的价值已在当前 step 实现，因此其 delayed capacity 固定为 0。其他 operation family 使用冻结的、无训练的结构容量先验，并与当前 graph-state delayed signals 按 `0.75 / 0.25` 混合。全部先验均写入 YAML 配置，可用于后续消融，运行中不会跨题更新。

## 6. 冻结 trace 的零 API replay

在不改写原 artifact 的前提下，使用最终图 provenance 重算 v2.4/v2.4.1 的双时间尺度指标：

| 冻结运行 | allocations | immediate ρ | delayed ρ | choice delayed ρ | choice n |
|---|---:|---:|---:|---:|---:|
| v2.4 | 236 | 0.486 | 0.448 | 0.398 | 54 |
| v2.4.1 | 267 | 0.464 | 0.353 | 0.223 | 56 |

在 v2.4.1 的 13 个 proof-gap operations 中，4 个成功恢复操作的平均 delayed return 为 0.647，9 个失败操作为 0。该结果验证了新目标量能够表达 v2.4.1 诊断中观察到的 delayed benefit，但不构成 v2.4.2 在线策略的最终实验结论。

## 7. Hard gate 与停止规则

正式顺序为：全量测试 → 冻结 trace replay → 推送 source commit → Smoke-A20 → Shadow-B20 → matched controls/dev50。任一预注册 hard gate 失败立即安全停止，不打开下一阶段。总 campaign 上限为 2000 provider attempts 或 2,000,000 provider-reported tokens，任一先达到即停止。

完整阈值见 `configs/dynamic_v242_preregistration.json`。
