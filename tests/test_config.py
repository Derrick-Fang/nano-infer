import json

from python.model.qwen3 import Qwen3Config


def test_from_pretrained_reads_fields(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 1000,
        "rope_theta": 500000.0,
        # unrelated fields (e.g. the transformers version) should be ignored
        "transformers_version": "5.0.0",
    }))
    cfg = Qwen3Config.from_pretrained(tmp_path)
    assert cfg.hidden_size == 64
    assert cfg.head_dim == 32
    assert cfg.num_key_value_heads == 2
    assert cfg.rope_theta == 500000.0


def test_head_dim_defaults_to_hidden_over_heads(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 64,
        "num_attention_heads": 4,
    }))
    cfg = Qwen3Config.from_pretrained(tmp_path)
    assert cfg.head_dim == 16
