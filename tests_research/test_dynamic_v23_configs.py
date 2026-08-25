from dataclasses import asdict
import json
from pathlib import Path

from tdca_research.dynamic_v2.config import DynamicV2ResearchConfig


def test_v23_smoke_controls_are_valid_and_mechanism_matched():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "configs/dynamic_hypergraph_v23_qwen_smoke20.yaml",
        root / "configs/dynamic_hypergraph_v23_qwen_smoke20_uniform.yaml",
        root / "configs/dynamic_hypergraph_v23_qwen_smoke20_fixed.yaml",
    ]
    configs = [DynamicV2ResearchConfig.from_yaml(path) for path in paths]
    for config in configs:
        config.validate()
        assert config.terminal_dependency_closure
        assert config.focused_empty_extraction_recovery
        assert config.goal_conditioned_join_frontier
        assert config.prompt_version == "dynamic-hypergraph-v2.3.3-structural-recovery"
    payloads = []
    for config in configs:
        payload = asdict(config)
        payload.pop("allocator_mode")
        payloads.append(payload)
    assert payloads[0] == payloads[1] == payloads[2]
    assert {config.allocator_mode for config in configs} == {
        "adaptive_evc", "uniform", "fixed_order",
    }


def test_v241_config_and_shadow_split_are_frozen_and_disjoint():
    root = Path(__file__).resolve().parents[1]
    config = DynamicV2ResearchConfig.from_yaml(
        root / "configs/dynamic_hypergraph_v241_qwen_smoke20.yaml"
    )
    config.validate()
    assert config.proof_gap_conditioned_recovery
    assert config.proof_usable_target_gate
    assert config.feasibility_reasoned_recovery
    assert config.no_diff_editor_preallocation_gate
    assert config.choice_conditioned_evc
    assert config.campaign_provider_call_cap == 2000
    assert config.campaign_provider_token_cap == 2_000_000

    base = json.loads((
        root / "configs/splits/musique_dynamic_seed20260820.json"
    ).read_text(encoding="utf-8"))
    shadow = json.loads((
        root / "configs/splits/musique_dynamic_v241_shadow_seed20260820.json"
    ).read_text(encoding="utf-8"))
    frozen_ids = {
        qid for ids in base["splits"].values() for qid in ids
    }
    shadow_ids = shadow["splits"]["smoke"]
    assert len(shadow_ids) == len(set(shadow_ids)) == 20
    assert frozen_ids.isdisjoint(shadow_ids)
    hop_counts = {
        hop: sum(qid.startswith(f"{hop}hop") for qid in shadow_ids)
        for hop in (2, 3, 4)
    }
    assert hop_counts == {2: 6, 3: 9, 4: 5}
