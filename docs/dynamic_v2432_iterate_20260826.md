# Dynamic Hypergraph TDCA v2.4.3.2：Smoke-A 结果与自动迭代决策

日期：2026-08-26
训练：无
模型：Qwen-plus
数据：MuSiQue distractor，冻结 Smoke-A20，seed = 20260820
源代码冻结：`c74c127994c465dc0cbf4810eb878282beb1d58c`

## 1. 本轮机制

v2.4.3.2 将 proof-obligation closure 与 execution-state transition 分离：

$$
V(o\mid G)=w_iV_{\mathrm{immediate}}+w_cV_{\mathrm{closure}}
+V_{\mathrm{transition}}-C_{\mathrm{absolute}}.
$$

其中认证转移价值为：

$$
V_{\mathrm{transition}}
=P_{\mathrm{cert}}\,\Delta R_{\mathrm{successor}}\,V_{\mathrm{option}}(G')
-R_{\mathrm{transition}}.
$$

只有由当前 sealed graph 重新验证、确定性、零 provider 调用且保证改变执行状态的
`commit:default` 或 assignment materialization 才能在通用净 EVC 阈值之前词典序执行。
控制器随后审计承诺的状态差分是否真正发生。

历史 v2.4.3.1 trace 的 gold-free 反事实回放表明：16 个阈值停止中有 15 个是可认证的
零 provider transition，全部被新规则解锁；另 1 个无证书操作仍保持停止。

## 2. Smoke-A20 结果

运行目录：
`research_outputs/musique_distractor_smoke_dynamic_hypergraph_tdca_v2_1787726403204360074`

主要结果：

| 指标 | v2.4.3.1 | v2.4.3.2 | Gate |
|---|---:|---:|---:|
| Candidate presence | 0.20 | 0.55 | 0.75 |
| Execution-plan completion | 0.10 | 0.60 | 0.75 |
| Graph-proof completion | 0.25 | 0.80 | 0.80 |
| F1 | 0.07 | 0.495 | 0.58 |
| Answered rate | 0.10 | 0.60 | — |
| Selective accuracy | 0.50 | 0.75 | — |
| Delayed EVC Pearson | 0.078 | 0.161 | 0.15 |
| Choice-conditioned delayed Pearson | -0.172 | 0.041 | $>0$ |
| Immediate EVC Pearson | -0.359 | -0.019 | 0.10 |

本轮 20/20 artifact 完整，125 provider attempts、151,057 provider-reported tokens，零基础设施失败、
零 invariant violation、零 controller-only mutation violation、零 unsupported answer、零 infeasible JOIN。
40 个认证状态转移全部真实兑现；评估器最初报告的 17 个 invalid bypass 是 allocation ID
跨题复用造成的索引错误，改为 `(qid, allocation_id)` 联合键后真实计数为 0。

## 3. 失败诊断

结果说明 transition 缺口已经修复：graph proof、delayed calibration 和 12 个 terminal readout
均大幅改善。剩余问题集中在 retrieval/extraction 的状态语义，而不是停止阈值。

控制器历史逻辑会在每次 RETRIEVE 后执行：

```text
subgoal.instantiated_question = retrieval_query
```

proof-gap recovery query 常采用 “Independent source for …” 形式。它因此覆盖了真正的推理子问题，
导致后续 extractor 面对的是检索指令而非目标关系。三个 4-hop dead end 的 extraction 均返回
`raw_claim_count = 0`；最终图中 `instantiated_question` 也保留了 recovery query。该问题与具体答案无关，
属于 reasoning objective 和 retrieval expression 混用的通用状态建模错误。

Immediate calibration 的离线分量审计也显示，旧 immediate 组合把 difficulty/heat 和历史 prior
误当作一步进展。239 个 allocation 上，五个归一化的一步进展分量等权组合与 actual immediate
utility 的 Pearson 为 0.467，而旧组合为 -0.019。

## 4. v2.4.3.3 自动迭代边界

下一轮只采用两个一般性修改：

1. RETRIEVE 仅把 query 写入 retrieval/evidence ledger，不再改变 subgoal reasoning objective；
2. Immediate value 改为已归一化的 evidence novelty、obligation closure、operation-conditioned
   closure mass、terminal gap 和 answer impact 的等权加和。

Meta-stop threshold、terminal acceptance gate、JOIN feasibility gate、训练设置和所有安全 invariant
保持不变。若 v2.4.3.3 仍未通过，将继续以新 trace 做结构诊断，不运行 Shadow-B，也不通过改单题或
放宽 gate 获得表面提升。
