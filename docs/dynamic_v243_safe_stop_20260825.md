# Dynamic Hypergraph TDCA v2.4.3 Smoke-A Safe-stop

日期：2026-08-25
决策：**SAFE_STOP**
Shadow-B：未运行
v2.4.4：未进入
训练：无

## 1. 冻结身份

- Source-freeze commit：`1dd422030e6bfb19932095565d14948e903fee2e`
- Run：
  `research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787657052732927059`
- Run manifest source tree：
  `fa21844ad227bebaab715e46d29a525df9e73e17e9dd947af30ca6145f950834`
- Seed：20260820
- Model：Qwen-plus
- Artifact：20/20，checksum 完整，infrastructure failure = 0
- Provider usage：176 attempts / 180628 reported tokens
- v2.4.3 campaign remaining：1695 attempts / 1677721 reported tokens
- v2.4.2 + v2.4.3 combined usage：305 attempts / 322279 reported tokens

## 2. Gate 结论

v2.4.3 通过了 23 项检查，失败 8 项，因此按照预注册规则停止。

### 2.1 通过的关键硬门

- zero leakage；
- zero infrastructure failure；
- zero graph invariant violation；
- controller-only mutation；
- zero unsupported ANSWER；
- zero selected infeasible JOIN；
- zero repeated same-fingerprint extraction；
- zero no-diff editor allocation；
- graph proof completion = 0.80；
- immediate EVC correlation = 0.4873；
- successful proof-gap delayed return = 0.6887；
- failed proof-gap delayed return = 0.0049；
- complete EVC trace = 1.0；
- complete delayed-credit trace = 1.0；
- complete proof-obligation trace = 1.0；
- non-uniform allocation = 1.0；
- no-executable without certificate = 0；
- viable proof opportunity cost-clipping stop = 0；
- absolute-cost ready-set invariance = 1.0；
- all ABSTAIN outcomes have exhaustion evidence；
- ANSWER / ABSTAIN / BUDGET_EXHAUSTED 保持分离。

这说明 v2.4.3 的基础设施、绝对成本不变性、Proof Obligation ledger 和 Certified Meta-stop 实现本身有效。

### 2.2 失败项

| Gate | 要求 | v2.4.3 |
|---|---:|---:|
| candidate presence | ≥ 0.75 | 0.65 |
| execution plan completion | ≥ 0.75 | 0.60 |
| F1 | ≥ 0.58 | 0.4643 |
| logical LLM calls | ≤ 163 | 184 |
| logical tokens | ≤ 185000 | 186981 |
| budget exhaustion rate | ≤ 0.10 | 0.25 |
| delayed EVC correlation | ≥ 0.15 | -0.1210 |
| choice-conditioned delayed correlation | > 0 | -0.2670 |

终止分布：

- ANSWER：12；
- ABSTAIN：3；
- BUDGET_EXHAUSTED：5。

## 3. 与冻结版本的比较

| 指标 | v2.4.1 | v2.4.2 | v2.4.3 |
|---|---:|---:|---:|
| EM | 0.55 | 0.55 | 0.45 |
| F1 | 0.5843 | 0.5643 | 0.4643 |
| candidate presence | 0.75 | 0.75 | 0.65 |
| plan completion | 0.75 | 0.65 | 0.60 |
| graph proof completion | 0.80 | 0.75 | 0.80 |
| answered rate | 0.75 | 0.65 | 0.60 |
| selective accuracy | 0.7333 | 0.8462 | 0.75 |
| logical calls | 148 | 135 | 184 |
| logical tokens | 167857 | 146230 | 186981 |
| budget exhaustion | 0.05 | 0.00 | 0.25 |

相对 v2.4.2：

- chain retained：10；
- chain gained：2；
- chain lost：3；
- absent in both：5；
- candidate presence gained：0；
- candidate presence lost：2；
- logical calls：+49；
- logical tokens：+40751；
- retrieval calls：-3。

chain gained：

- `3hop1__801799_547811_41132`；
- `4hop1__152146_5274_458768_33637`。

chain lost：

- `3hop1__105767_443779_52195`；
- `3hop2__326964_7845_7713`；
- `4hop1__107309_457883_650651_7262`。

candidate presence lost：

- `3hop1__145924_131905_41948`；
- `4hop1__107309_457883_650651_7262`。

这些 qid 只用于完成后的配对诊断，没有进入 controller、prompt、配置或运行时分支。

## 4. 核心诊断

### 4.1 绝对成本修复了误裁剪，但没有正确计价 fidelity

离线 replay 已证明新成本消除了上一轮两个“预算充足却被 min-max 成本置零”的情况，实际 Smoke 中：

- viable proof opportunity cost-clipping stop = 0；
- ready-set cost invariance = 1；
- `4hop1__51465_53706_795904_580996` 不再因旧成本公式直接置零。

但是高 fidelity 的实际调用数没有被完整计入 predicted call cost。尤其 VERIFY：

| VERIFY budget | v2.4.2 | v2.4.3 |
|---|---:|---:|
| selected count | 55 | 56 |
| mean max tokens | 474.4 | 900 |
| mean verification samples | 1.0 | 2.0 |
| high-fidelity count | 0 | 56 |

v2.4.3 的 VERIFY 数量只增加 1 次，但所有 VERIFY 都从单样本升级成双样本/900 tokens，直接解释了大量额外 calls 和 tokens。

因此问题不是“operation 数量无界增加”，而是：

> fidelity 的边际收益被放大，而额外 verification sample 的边际调用成本没有被准确计价。

### 4.2 混淆了 obligation importance 与 closure probability

当前 delayed value 近似把严重 proof gap 当成高预期回报，但：

[
	ext{important gap}
otRightarrow
	ext{current operation can close the gap}.
]

按 target obligation 聚合：

| Obligation | N | Predicted delayed | Actual delayed | Gap |
|---|---:|---:|---:|---:|
| contradiction | 2 | 0.666 | 0.000 | +0.666 |
| missing binding | 6 | 0.796 | 0.090 | +0.706 |
| missing claim | 53 | 0.511 | 0.355 | +0.155 |
| missing evidence | 48 | 0.347 | 0.478 | -0.131 |
| missing JOIN premise | 17 | 0.843 | 0.423 | +0.420 |
| missing verification | 56 | 0.490 | 0.374 | +0.115 |

结论：

- missing binding 和 missing JOIN premise 被严重高估；
- missing evidence 反而被低估；
- contradiction 样本少，但当前操作没有产生 delayed return；
- 有 7 个 allocation 满足 predicted delayed ≥ 0.70、actual delayed ≤ 0.10。

### 4.3 Operation-to-obligation 映射过宽

第一版映射允许 BRANCH/extraction 声称关闭 `missing_binding`。但 extraction 的局部 mutation 不等于依赖绑定完成；同理：

- VERIFY mutation 不保证 terminal proof 前沿前进；
- JOIN validation 成功不保证新 conclusion 被终局证明使用；
- COMMIT subgoal 不保证后续依赖仍可执行；
- `progressed=True` 只表示图发生 mutation，不等于 obligation 被因果关闭。

因此下一版必须基于 operation 后的 obligation delta 和 causal descendant，而不是只根据 operation family 声明可关闭类型。

### 4.4 Terminal reachability 区分力不足

在同一可达 subgoal 内，terminal reachability 往往是常数。它可以排除完全断开的区域，却无法区分：

- 新颖证据与重复证据；
- 可行 JOIN 与冗余 JOIN；
- 能关闭 binding 的 extraction 与只增加候选的 extraction；
- 必要 verification 与重复 verification。

这解释了 graph-local feature 在总体上语义合理，但 within-family delayed correlation 为负。

### 4.5 Proof graph 完整不等于候选质量安全

graph proof completion 从 v2.4.2 的 0.75 恢复到 0.80，但 candidate presence 从 0.75 降至 0.65，最终 F1 下降。说明 Proof Obligation 机制能够组织和审计已有图，却没有保证正确候选被生成并保留。

## 5. 下一阶段边界

本轮不能进入 v2.4.4 diffusion。传播一个负相关的 allocation signal 会放大错误，而不是修复错误。

建议下一轮为 **v2.4.3.1：Operation-conditioned Obligation Closure**，仅修复 v2.4.3 policy：

[
V_{mathrm{delayed}}(o)=
I(O)cdot
P_{mathrm{close}}(Omid o,G)cdot
Delta O(o)cdot
R_{mathrm{terminal}}(O)
-
R_{mathrm{redundancy}}(o).
]

其中保持无训练：

- (I(O))：obligation importance；
- (P_{mathrm{close}})：由 deterministic feasibility、premise completeness、previous same-region failure、novelty 和 type constraints 得出；
- (Delta O(o))：该操作实际能减少的 obligation channel，而非 family 声明；
- (R_{mathrm{terminal}})：终局可达性；
- (R_{mathrm{redundancy}})：重复 verification/retrieval/JOIN 风险。

成本必须按 fidelity 的真实资源需求计算：

- VERIFY low/medium/high 的 expected call demand 分别匹配 verification samples；
- high fidelity 必须用“相对 medium 的边际收益”减“相对 medium 的边际成本”；
- reserved minimum calls/tokens 应随尚未关闭的 critical obligations 增加；
- 不回退到 ready-set normalization；
- 不简单降低 meta-stop threshold 或手工提高单一 cost weight。

必须冻结保留：

- absolute ready-set invariance；
- Proof Obligation ledger 与 snapshots；
- Certified Meta-stop certificate；
- horizon-aware causal credit target；
- controller-only mutation；
- independent terminal belief hard gate；
- zero unsupported ANSWER。

只有 v2.4.3.1 在冻结 trace replay 中恢复正的 within-family ranking，并重新通过 Smoke-A 与 Shadow-B，才允许进入 v2.4.4 signed hypergraph diffusion。
