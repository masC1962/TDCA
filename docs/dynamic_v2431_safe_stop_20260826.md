# TDCA v2.4.3.1 Smoke-A SAFE_STOP 报告

## 1. 结论

v2.4.3.1 完成源码冻结、零 API 回放、257 项测试和固定 MuSiQue Smoke-A 20。Artifact 20/20 完整，安全与审计门槛全部通过，VERIFY 双样本成本漏计和高保真坍塌均已修复，但质量及 EVC calibration 严重失败。因此本轮结论为 **SAFE_STOP**：不运行 Shadow-B，不进入 signed diffusion，不修改既定阈值追逐这 20 题。

本轮最重要的科学结论是：

> Proof-obligation closure value 与 reasoning-state transition / option value 是两个不同的量。仅估计“当前操作关闭当前 obligation 的能力”，会错误停止那些不直接关闭 obligation、但确定性解锁下游推理的操作。

## 2. 冻结信息

- Source-freeze commit：`fd7b333574bfc73ec50611fb7d54a4747b063326`
- Smoke-A run：`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787720103222432618`
- 固定 seed：`20260820`
- 模型：Qwen-plus，temperature 0
- 训练：无
- Smoke 独立安全帽：250 provider attempts / 250,000 provider-reported tokens
- 实际用量：87 attempts / 94,372 provider-reported tokens，pending = 0
- 研究累计用量：392 attempts / 416,651 provider-reported tokens
- 全局剩余额度：1,608 attempts / 1,583,349 provider-reported tokens

## 3. 零 API 阶段

冻结 v2.4.3 的 20 个 graph artifact 和 239 个 selected allocation 上，gold-free replay 得到：

- 旧 operation–obligation 过宽声明：8 次；
- 旧 VERIFY 双样本被预测为单调用：56 次；
- 旧 delayed prediction 与 target closure Pearson：-0.1945；
- replay decision：`GO_SOURCE_FREEZE_AND_SMOKE_A`。

全量测试为 257/257 通过。测试覆盖 strict targeting、importance/tractability 分离、exact sample accounting、positive marginal high-fidelity gate、budget reserve、controller-owned actual closure 和版本化 seal roundtrip。

## 4. Smoke-A gate

| 指标 | v2.4.3 | v2.4.3.1 | Gate | 结果 |
|---|---:|---:|---:|---|
| Candidate presence | 0.65 | 0.20 | ≥ 0.75 | FAIL |
| Execution plan completion | 0.60 | 0.10 | ≥ 0.75 | FAIL |
| Graph proof completion | 0.80 | 0.25 | ≥ 0.80 | FAIL |
| F1 | 0.4643 | 0.0700 | ≥ 0.58 | FAIL |
| Logical LLM calls | 184 | 88 | ≤ 163 | PASS |
| Logical tokens | 186,981 | 94,874 | ≤ 185,000 | PASS |
| Budget exhaustion | 0.25 | 0.00 | ≤ 0.10 | PASS |
| Immediate EVC Spearman | 0.4873 | -0.3595 | ≥ 0.10 | FAIL |
| Delayed EVC Spearman | -0.1210 | 0.0778 | ≥ 0.15 | FAIL |
| Choice delayed correlation | -0.2670 | -0.1716 | > 0, n≥20 | FAIL |

Termination 为 ANSWER 2 / ABSTAIN 18 / BUDGET_EXHAUSTED 0。EM 为 0.05，answered rate 为 0.10，selective accuracy 为 0.50。

以下门槛全部通过：

- zero leakage、zero infrastructure failure；
- zero invariant violation、controller-only mutation；
- unsupported ANSWER = 0；
- infeasible JOIN = 0；
- repeated extraction fingerprint = 0；
- no-diff editor allocation = 0；
- complete EVC / delayed credit / proof obligation trace = 1.0；
- non-uniform allocation = 1.0；
- 所有 ABSTAIN 均有 exhaustion/dead-end certificate；
- predicted provider calls 与 requested samples 一致率 = 1.0；
- marginal EVC≤0 的 high-fidelity allocation = 0；
- operation target audit 和 actual closure trace 完整率 = 1.0。

## 5. 主因诊断

### 5.1 Certified COMMIT 被错误地按 closure operation 估值

18 个 ABSTAIN 中：

- 15 个为 `certified_commit_below_threshold`；
- 2 个为 `no_executable_with_certificate`；
- 1 个为其他 below-threshold stop；
- 17 个停止时仍保有至少 4 calls 和 4,000 tokens。

`commit:default` 已通过确定性可提交证书，不消耗 provider call，并会把已证实的中间结论写入 branch assignment，从而暴露下游 subgoal。它不直接关闭当前 obligation，因此 v2.4.3.1 给出：

$$
I=P_{\mathrm{close}}=\Delta O=R_{\mathrm{terminal}}=0,
\qquad V_{\mathrm{delayed}}=0.
$$

该 family 的平均 immediate prediction 为 0.3493，经过全局 0.40 immediate / 0.60 delayed 混合并扣除 0.00325 成本后，许多 COMMIT 的 net EVC 低于 0.08。系统因此在一个必要、确定性、近零成本的状态转移之前 ABSTAIN。

这不是阈值需要降低，而是 value taxonomy 错误：COMMIT 的价值属于 **certified transition value**，而不是 obligation closure value。

### 5.2 Closure 与 downstream return 使用了不同监督目标

各 family 的实际 target closure rate 很高：

- RETRIEVE：0.9412；
- EXTRACT：0.9394；
- VERIFY：1.0000；
- MERGE：0.7500。

说明 strict operation–obligation contract 本身有效。但 gate 把 closure-based prediction 与 provenance delayed proof return 做相关性比较；两者并非同一个随机变量。例如 `commit:default` 的 target closure 为 0，却有平均 0.2023 的 downstream delayed return；`branch:assignments` closure 为 0，却有平均 0.5731 delayed return。

下一版必须分别校准：

1. `closure prediction` 对 `actual target obligation delta`；
2. `transition/option prediction` 对 `causal downstream return`。

二者可在最终 EVC 中相加，但不可在建模和评估阶段混为一个量。

### 5.3 Multiplicative closure 过早截断推理 horizon

当前公式同时乘以 obligation mass 占比与 terminal distance。当操作首先建立中间状态、其 successor obligation 尚未显式出现时，它会被重复折扣。虽然 RETRIEVE 的平均 target closure 达 0.9412，系统仍可能在随后的确定性 COMMIT 处停止，导致整条 chain 无法展开。

与 v2.4.3 配对比较：candidate presence 丢失 9 题、graph proof 丢失 11 题，无任何 candidate/proof gain。allocation 总体减少 117 次，其中 EXTRACT -25、RETRIEVE -23、VERIFY -23、COMMIT -37、MERGE -9。这是推理 horizon 被截断，而不是仅仅“更高效”。

## 6. 本轮真正修复成功的内容

以下机制应冻结保留：

- strict operation–obligation targeting；
- obligation importance 与 closure tractability 分离；
- controller-owned pre/post obligation closure trace；
- VERIFY sample-exact call/token accounting；
- VERIFY 全部从双样本 high fidelity 降为单样本 medium，且无 unjustified high fidelity；
- critical-obligation reserve certificate；
- absolute ready-set-invariant cost；
- controller-only mutation、JOIN feasibility、certified stop 和终止类型分离。

成本从 v2.4.3 的 184 calls / 186,981 logical tokens 降至 88 / 94,874，但该点不是有效 Pareto improvement，因为质量显著下降。

## 7. 下一轮建议：v2.4.3.2 Certified Transition and Option Value

暂不进入 v2.4.4 diffusion。建议把总价值拆为三个独立、可审计通道：

$$
V(o)=V_{\mathrm{immediate}}(o)
+V_{\mathrm{closure}}(o)
+V_{\mathrm{transition}}(o)
-C_{\mathrm{abs}}(o).
$$

其中：

$$
V_{\mathrm{transition}}(o)
=P_{\mathrm{cert}}(o\mid G)
\cdot \Delta R_{\mathrm{successor}}(o)
\cdot V_{\mathrm{option}}(G').
$$

下一轮的边界应为：

1. 已通过证书、确定性、零 provider-call 且会推进 branch state 的 COMMIT/assignment 属于 mandatory transition，不受通用 delayed threshold 阻断；
2. `V_transition` 只来自可证明的 successor-state diff，不恢复 operation-family 常数；
3. closure calibration 与 downstream transition calibration 分开报告；
4. 使用冻结 v2.4.3.1 trace 和 synthetic graph fixtures 做零 API replay；
5. 不降低 `meta_stop_evc_threshold`，不改质量 gate，不加入题目补丁；
6. 新 Smoke 前先证明：15 个 certified-COMMIT false stop 在反事实中被解除，同时不可执行操作仍会停止。

只有 v2.4.3.2 重新通过当前 Smoke-A 全部门槛后，才授权 Shadow-B；signed diffusion 继续冻结。
