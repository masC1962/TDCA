import json
from pathlib import Path

from tdca_research.config import ResearchConfig
from tdca_research.data import load_examples, select_split
from tdca_research.utils import sha256_file


FROZEN_CONFIGS = (
    "configs/structured_tdca_qwen_validation200_frozen.yaml",
    "configs/structured_tdca_qwen_hotpot_smoke20_frozen.yaml",
    "configs/structured_tdca_qwen_hotpot_tuning50_frozen.yaml",
    "configs/structured_tdca_qwen_2wiki_smoke20_frozen.yaml",
    "configs/structured_tdca_qwen_2wiki_tuning50_frozen.yaml",
)


def test_frozen_configs_reference_present_manifests_and_exact_split_sizes():
    expected = {"smoke": 20, "tuning": 50, "validation": 200}
    for path in FROZEN_CONFIGS:
        config = ResearchConfig.from_yaml(path)
        manifest_path = Path(config.split_manifest_path)
        assert manifest_path.exists(), path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("dataset_sha256") == sha256_file(config.dataset_path), path
        examples = load_examples(config.dataset_path, config.dataset)
        selected = select_split(examples, config.split, manifest, config.split_seed)
        assert len(selected) == expected[config.split], path
        assert len({example.qid for example in selected}) == len(selected)


def test_every_frozen_manifest_is_disjoint():
    manifests = {
        ResearchConfig.from_yaml(path).split_manifest_path
        for path in FROZEN_CONFIGS
    }
    for manifest_path in manifests:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        split_sets = [set(ids) for ids in manifest["splits"].values()]
        for index, left in enumerate(split_sets):
            for right in split_sets[index + 1 :]:
                assert left.isdisjoint(right), manifest_path
