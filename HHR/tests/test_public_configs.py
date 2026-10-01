import configparser
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_configs():
    for name in ("llama31_sparse.ini", "qwen3_4b_sparse.ini"):
        config = configparser.ConfigParser()
        config.read(ROOT / "configs" / name)
        assert config.getfloat("dataset", "TOPK_RATIO") == 0.015
        assert config.getint("method", "NUM_SINK") == 2
        assert config.getint("method", "NUM_RECENT") == 0
        assert config.getfloat("quest", "RATIO") == 0.30
        assert config.getint("quest", "PAGE_SIZE") == 8


def test_head_budget():
    payload = json.loads(
        (ROOT / "configs" / "head_budget_q30_page8.json").read_text()
    )
    assert payload["head_axis"] == "kv_head"
    assert payload["page_size"] == 8
    assert payload["sparse_ratio"] == 0.015
    assert payload["default"] == [0.3] * 8


if __name__ == "__main__":
    test_sparse_configs()
    test_head_budget()
    print("public config tests passed")
