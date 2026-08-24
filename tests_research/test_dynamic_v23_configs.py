from dataclasses import asdict
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
