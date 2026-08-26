# TDCA v2.4.3.1：Operation-conditioned Obligation Closure

## 1. 研究动机

v2.4.3 Smoke-A 的安全机制全部正常，但质量与成本门槛失败。其 delayed-value 与真实 delayed return 的相关性为负，且 56 次 VERIFY 全部选择了双样本高保真。诊断显示，旧策略把 proof obligation 的严重程度近似当作当前 operation 的可关闭性，并按一次调用预测 VERIFY 成本。因此，高价值但当前不可关闭的 obligation 会获得过高分，双样本验证则被系统性低估。

v2.4.3.1 不改变检索器、提示词任务语义、JOIN 规则、终止证书或答案门槛。它只修复 obligation closure 与 fidelity accounting 两条决策路径，并保持无训练、gold-free 和 controller-only mutation。

## 2. 价值函数

对候选操作 \(o\)、图状态 \(G\) 和其严格指向的 proof obligation 集合 \(O\)，定义：

$$
V_{\mathrm{delayed}}(o)=
I(O)\,P_{\mathrm{close}}(O\mid o,G)\,
\Delta O(o)\,R_{\mathrm{terminal}}(O)
-R_{\mathrm{redundancy}}(o).
$$

各分量保持独立记录，不压缩成一个不可审计的概率：

- \(I(O)\)：obligation severity，只表示重要性；
- \(P_{\mathrm{close}}\)：由具体 operation payload、前提完整性、证据覆盖、类型约束和相同区域失败决定；
- \(\Delta O\)：该 operation 可减少的目标 obligation mass 占区域 open mass 的比例；
- \(R_{\mathrm{terminal}}\)：目标 obligation 到 terminal objective 的依赖距离；
- \(R_{\mathrm{redundancy}}\)：重复 query、重复 JOIN signature 或相同区域失败的独立惩罚，开发配置上限为 0.15。

## 3. 严格操作契约

| Operation | 可声明关闭的 obligation | 必要条件 |
|---|---|---|
| RETRIEVE | `missing_evidence` | 非空、具新颖性的 query，仍有 retrieval capacity |
| BRANCH/extract | `missing_claim` | obligation 要求的 evidence 全部位于 operation sources |
| VERIFY | `missing_verification` | 所有目标 claim 均为 sources，且具有 grounding evidence |
| MERGE/JOIN | `missing_join_premise` | 至少两个有效 claim、JOIN signature、确定性约束 payload |
| REVISE/PRUNE | `contradiction` | 明确覆盖发生 contradiction 的 claim |
| EXPAND | `terminal_disconnected_join` | 明确的 terminal-path graph-edit event |
| COMMIT | 无 delayed closure | 只实现已经建立的 terminal value |

`missing_binding` 不再由泛化的 BRANCH/EXPAND 自动声明关闭。Binding 是 branch-local state；只有显式 branch assignment 才视为依赖已绑定。

## 4. Fidelity 与预算

VERIFY 的 predicted provider calls 必须等于 `verification_samples`。Token upper bound 为每次请求 token cap 与样本数的乘积。对 medium 到 high 的升级，使用：

$$
\mathrm{mEVC}_{m\rightarrow h}
=\Delta \mathrm{GrossValue}_{m\rightarrow h}
-\Delta \mathrm{AbsoluteCost}_{m\rightarrow h}.
$$

仅当 marginal EVC 严格为正，并且升级后仍保留关键 terminal-reachable obligation 的最小 call/token reserve 时，high fidelity 才进入可执行 ready set。所有预测调用数、token upper bound、marginal EVC 和 reserve certificate 均写入 allocation trace。

## 5. 真实 closure 审计

Controller 在 operation 执行前记录目标 obligation 状态，在 mutation、belief update 和 obligation refresh 后记录：

- `actual_closed_target_ids`；
- `actual_target_closure_rate`；
- `actual_obligation_delta`。

Graph mutation 仍只允许通过 `V2GraphController`。历史 v2.4.3 artifact 的 seal 通过版本化序列化保持兼容。

## 6. 零 API 回放

冻结 v2.4.3 的 20 个 graph-state artifact 包含 239 个已选择 allocation。Gold-free 回放发现：

- 8 个 allocation 使用了过宽的 operation–obligation 声明；
- 56 个 VERIFY allocation 的旧成本模型把双样本预测为一次调用；
- old predicted delayed value 与 target closure 的 Pearson 相关系数为 -0.1945。

回放不读取 dataset、prediction、answer 或 metric 文件，也不宣称历史反事实质量提升。它只验证缺陷确实存在；新策略的可执行性、不变量和严格契约由 deterministic tests 验证。

## 7. 实验顺序与停止条件

1. 全量零 API 测试；
2. 冻结 v2.4.3 graph trace 回放；
3. 提交并推送 source-freeze commit；
4. 固定 MuSiQue Smoke-A 20；
5. 任一 preregistered hard gate 失败即 SAFE_STOP；
6. 只有 Smoke-A 全部通过才授权 Shadow-B 20。

Smoke-A 独立上限为 250 provider attempts / 250,000 provider-reported tokens。全研究累计上限仍为 2000 / 2,000,000。禁止修改门槛追逐样本、禁止题号/答案补丁，暂不启用 signed diffusion。
