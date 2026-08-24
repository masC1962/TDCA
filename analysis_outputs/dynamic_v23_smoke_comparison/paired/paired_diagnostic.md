# Dynamic Hypergraph TDCA paired chain diagnostic

- Provider/LLM calls made: 0
- Chain gained: `['3hop1__140786_2053_5289', '3hop2__326964_7845_7713']`
- Chain lost: `['3hop1__145924_131905_41948', '3hop2__90327_87184_76291', '4hop1__51465_53706_795904_580996']`

## 3hop1__145924_131905_41948

- Status/calls: answer/7 -> answer/6
- Allocation families: `{'retrieve:default': 3, 'branch:extract_typed': 3, 'verify:default': 3, 'commit:default': 3, 'merge:validate_join': 2, 'commit:answer': 1}` -> `{'retrieve:default': 3, 'branch:extract_typed': 2, 'verify:default': 2, 'commit:default': 1, 'commit:answer': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}`
- Extraction: `[{'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 3, 'accepted_rows': 3, 'rejections': {}}, {'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}]`
- JOIN rejection reasons: `{}`
- JOIN attempts: `[]`

## 3hop2__90327_87184_76291

- Status/calls: answer/11 -> budget_exhausted/15
- Allocation families: `{'retrieve:default': 5, 'branch:extract_typed': 4, 'verify:default': 6, 'commit:default': 3, 'merge:validate_join': 6, 'commit:answer': 1}` -> `{'retrieve:default': 5, 'branch:extract_typed': 4, 'verify:default': 6, 'commit:default': 2, 'merge:validate_join': 6}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'CONTINUE', 'reason': 'positive_expected_value', 'best_predicted_evc': 2.6920478361556963}`
- Extraction: `[{'target_id': 'subgoal_2', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 4, 'accepted_rows': 2, 'rejections': {'duplicate_triple': 1, 'ungrounded': 1}}, {'target_id': 'subgoal_root', 'focus_mode': 'direct_answer', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}]`
- JOIN rejection reasons: `{'operation_produced_no_commit': 2, 'StructuredOutputError': 1}`
- JOIN attempts: `[{'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'conjunctive_relational_path', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': True, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'conjunctive_relational_path', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': True, 'answer_used': False, 'reason': 'StructuredOutputError'}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': False, 'charged': True, 'answer_used': False, 'reason': 'operation_produced_no_commit'}]`

## 4hop1__51465_53706_795904_580996

- Status/calls: answer/7 -> abstain/3
- Allocation families: `{'retrieve:default': 3, 'branch:extract_typed': 3, 'verify:default': 3, 'commit:default': 3, 'merge:validate_join': 2, 'commit:answer': 1}` -> `{'retrieve:default': 2, 'branch:extract_typed': 1, 'expand:default': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'ABSTAIN', 'reason': 'no_executable_computation', 'best_predicted_evc': 0.0}`
- Extraction: `[{'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}]`
- JOIN rejection reasons: `{}`
- JOIN attempts: `[]`
