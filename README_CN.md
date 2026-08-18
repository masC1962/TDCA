# TDCA 适配版（当前代码基础上增强）

这版不是重写，而是在你现有 TDCA 原型上做增强，重点补三件事：

1. 让 `state` 真正展开成多步子状态，而不是只有根问题挂证据。
2. 让热扩散更偏向 **state 调度**，证据/记忆作为注入项而不是直接调度对象。
3. 避免 memory 直接写入答案泄漏，改成策略模板记忆。

## 新增到 data/ 的三个小子集

- `data/hotpotqa_subset_50.jsonl`
- `data/2wikimultihopqa_subset_50.jsonl`
- `data/musique_subset_50.jsonl`
- `data/dataset_manifest.json`

这些都是从公开 dev 集裁出来的小子集，方便你先做原型验证和 baseline 对比。

## 这次主要改了什么

- `tdca_scheduler.py`
  - 支持真实的多步 `sub_questions` 扩展
  - 加入 heuristic fallback，避免 LLM 没长出子状态时整棵图停住
  - 只对 `state/answer` 节点做热扩散，evidence/memory 作为注入项
  - 加入 duplicate merge，减少重复状态膨胀
  - trace 里增加 frontier snapshot，便于看调度是否真的发生

- `prompts.py`
  - 扩展 prompt 改成更强制的 TDCA 风格 JSON 输出
  - final answer 明确不输出 `<think>`

- `knowledge_memory.py`
  - memory 默认内容改成“策略模板”，不再默认泄漏答案
  - 写回 memory 时去重

- `llm_evaluator.py`
  - 清理 `<think>` 输出
  - MockLLM 跟新 prompt 对齐

- `core_models.py`
  - frontier 只扩 `state` 节点，不再扩 answer
  - 新增 `refines/verifies` 边类型

- `config.py`
  - 新增 `max_state_depth / support_reheat / duplicate_merge_gain / memory_write_min_value` 等参数

- `utils.py`
  - 新增 `normalize_text / lexical_jaccard / strip_think_blocks`

## 运行建议

先继续用你原来的方式跑：

```bash
python main.py --llm_backend openrouter --openai_base_url "$OPENROUTER_BASE_URL" --served_model_name "$OPENROUTER_MODEL" --openai_api_key "$OPENROUTER_API_KEY" --reasoning_effort none
```

如果要先看图是否真正长出多步状态，建议看：

- `outputs/.../trace.json`
- `outputs/.../graph.json`

重点看：
- `trace[].created_node_ids` 是否持续非空
- `steps` 是否 > 1
- `graph.json` 是否出现多个 `state_*` 以及 `state -> state` 边


## 使用 OpenRouter 的 GPT-4o 运行

当前代码已支持 `--llm_backend openrouter`，并通过 OpenAI 兼容接口调用 OpenRouter。

### 1）设置环境变量

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_MODEL="openai/gpt-4o"
export OPENROUTER_APP_NAME="TDCA"
# 可选
export OPENROUTER_SITE_URL="https://your-site.example"
```

### 2）单条问题运行

```bash
python main.py   --llm_backend openrouter   --openai_base_url "$OPENROUTER_BASE_URL"   --served_model_name "$OPENROUTER_MODEL"   --openai_api_key "$OPENROUTER_API_KEY"   --query "What is the birth city of the director of the movie Inception?"
```

### 3）HotpotQA 批量运行

```bash
bash run_hotpot_gpt4o_10.sh 10 data/hotpot_dev_distractor_v1.jsonl
```

### 4）说明

- `openrouter` 后端底层仍然走 OpenAI 兼容 SDK，但默认 URL 已改成 `https://openrouter.ai/api/v1`。
- 默认模型名已改为 `openai/gpt-4o`。
- 若你想固定别的模型，可直接改 `OPENROUTER_MODEL`。
