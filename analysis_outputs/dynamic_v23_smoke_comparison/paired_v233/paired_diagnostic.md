# Dynamic Hypergraph TDCA paired chain diagnostic

- Provider/LLM calls made: 0
- Chain gained: `['3hop1__140786_2053_5289', '3hop2__326964_7845_7713', '4hop2__103790_39078_8987_8529']`
- Chain lost: `['2hop__89764_827343', '3hop1__105767_443779_52195', '3hop1__132795_40769_64047', '3hop2__90327_87184_76291', '4hop1__51465_53706_795904_580996']`

## 2hop__89764_827343

- Status/calls: answer/9 -> abstain/9
- Allocation families: `{'retrieve:default': 3, 'branch:extract_typed': 4, 'verify:default': 4, 'commit:default': 2, 'merge:validate_join': 1, 'commit:answer': 1}` -> `{'retrieve:default': 3, 'branch:extract_typed': 4, 'verify:default': 3, 'commit:default': 1, 'merge:validate_join': 7, 'expand:default': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'ABSTAIN', 'reason': 'no_executable_computation', 'best_predicted_evc': 0.0}`
- Extraction: `[{'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 4, 'accepted_rows': 4, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'direct_answer', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}]`
- JOIN rejection reasons: `{'operation_produced_no_commit': 7}`
- JOIN attempts: `[{'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'shared_role', 'premises': 2, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'conjunctive_relational_path', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'shared_role_conjunction', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'conjunctive_relational_path', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}, {'kind': 'shared_role_conjunction', 'premises': 3, 'alignment': 0.55, 'accepted': False, 'charged': False, 'answer_used': False, 'reason': 'operation_produced_no_commit'}]`

## 3hop1__105767_443779_52195

- Status/calls: answer/7 -> budget_exhausted/8
- Allocation families: `{'retrieve:default': 3, 'branch:extract_typed': 3, 'verify:default': 3, 'commit:default': 3, 'merge:validate_join': 2, 'commit:answer': 1}` -> `{'retrieve:default': 5, 'branch:extract_typed': 3, 'verify:default': 5, 'commit:default': 2, 'merge:validate_join': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'CONTINUE', 'reason': 'positive_expected_value', 'best_predicted_evc': 2.071301828559828}`
- Extraction: `[{'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_2', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 5, 'accepted_rows': 4, 'rejections': {'ungrounded': 1}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 6, 'accepted_rows': 6, 'rejections': {}}]`
- JOIN rejection reasons: `{}`
- JOIN attempts: `[{'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}]`

## 3hop1__132795_40769_64047

- Status/calls: answer/8 -> abstain/8
- Allocation families: `{'retrieve:default': 4, 'branch:extract_typed': 4, 'verify:default': 3, 'commit:default': 3, 'merge:validate_join': 2, 'commit:answer': 1}` -> `{'retrieve:default': 3, 'branch:extract_typed': 5, 'verify:default': 1, 'commit:default': 1, 'expand:default': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'ABSTAIN', 'reason': 'no_executable_computation', 'best_predicted_evc': 0.0}`
- Extraction: `[{'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_2', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}, {'target_id': 'subgoal_2', 'focus_mode': 'direct_answer', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}, {'target_id': 'subgoal_2', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 1, 'accepted_rows': 0, 'rejections': {'ungrounded': 1}}, {'target_id': 'subgoal_2', 'focus_mode': 'direct_answer', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}]`
- JOIN rejection reasons: `{}`
- JOIN attempts: `[]`

## 3hop2__90327_87184_76291

- Status/calls: answer/11 -> budget_exhausted/15
- Allocation families: `{'retrieve:default': 5, 'branch:extract_typed': 4, 'verify:default': 6, 'commit:default': 3, 'merge:validate_join': 6, 'commit:answer': 1}` -> `{'retrieve:default': 5, 'branch:extract_typed': 6, 'verify:default': 6, 'commit:default': 2, 'merge:validate_join': 4, 'expand:default': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'CONTINUE', 'reason': 'positive_expected_value', 'best_predicted_evc': 2.621969999738366}`
- Extraction: `[{'target_id': 'subgoal_2', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 2, 'accepted_rows': 2, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': True, 'raw_rows': 4, 'accepted_rows': 2, 'rejections': {'duplicate_triple': 1, 'ungrounded': 1}}, {'target_id': 'subgoal_root', 'focus_mode': 'direct_answer', 'accepted': True, 'raw_rows': 1, 'accepted_rows': 1, 'rejections': {}}, {'target_id': 'subgoal_root', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 4, 'accepted_rows': 0, 'rejections': {'duplicate_triple': 3, 'ungrounded': 1}}, {'target_id': 'subgoal_dynamic_v2_20', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 1, 'accepted_rows': 0, 'rejections': {'duplicate_triple': 1}}]`
- JOIN rejection reasons: `{'operation_produced_no_commit': 1}`
- JOIN attempts: `[{'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'relational_path', 'premises': 2, 'alignment': 0.55, 'accepted': True, 'charged': True, 'answer_used': False, 'reason': ''}, {'kind': 'shared_role', 'premises': 2, 'alignment': 0.55, 'accepted': False, 'charged': True, 'answer_used': False, 'reason': 'operation_produced_no_commit'}]`

## 4hop1__51465_53706_795904_580996

- Status/calls: answer/7 -> abstain/4
- Allocation families: `{'retrieve:default': 3, 'branch:extract_typed': 3, 'verify:default': 3, 'commit:default': 3, 'merge:validate_join': 2, 'commit:answer': 1}` -> `{'retrieve:default': 2, 'branch:extract_typed': 2, 'expand:default': 1}`
- Terminal: `{'outcome': 'ANSWER', 'reason': 'accepted_graph_grounded_answer', 'best_predicted_evc': 0.0}` -> `{'outcome': 'ABSTAIN', 'reason': 'no_executable_computation', 'best_predicted_evc': 0.0}`
- Extraction: `[{'target_id': 'subgoal_1', 'focus_mode': 'coverage', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}, {'target_id': 'subgoal_1', 'focus_mode': 'direct_answer', 'accepted': False, 'raw_rows': 0, 'accepted_rows': 0, 'rejections': {}}]`
- JOIN rejection reasons: `{}`
- JOIN attempts: `[]`
