"""Verifies the model loads correctly with the real Qwen3-0.6B config +
weights.

Two levels of checking:
1. Completeness: every tensor in the HF weight files lands on a parameter
   of our model, and no parameter of ours is left unloaded
   (loaded == state_dict).
2. Numerics: for the same input, our model's logits match HF
   Qwen3ForCausalLM — much stronger evidence than "weights copied"
   (it validates QK-Norm, RoPE, GQA, packed concatenation, and everything
   else wired together).
"""

from pathlib import Path

import pytest
import torch

from python.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from python.model.loader import load_weights

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "Qwen3-0.6B"

needs_model = pytest.mark.skipif(
    not (MODEL_PATH / "model.safetensors").exists(),
    reason=f"local weights not found: {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def our_model():
    config = Qwen3Config.from_pretrained(MODEL_PATH)
    model = Qwen3ForCausalLM(config)
    loaded = load_weights(model, MODEL_PATH)
    # Completeness check: every parameter is loaded (the tied lm_head is
    # considered loaded by the loader)
    missing = set(model.state_dict()) - loaded
    assert not missing, f"parameters not loaded: {sorted(missing)}"
    return model.eval()


@needs_model
def test_config_parsed():
    cfg = Qwen3Config.from_pretrained(MODEL_PATH)
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 28
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (16, 8)
    assert cfg.head_dim == 128
    assert cfg.tie_word_embeddings is True
    assert cfg.attention_bias is False


@needs_model
def test_meta_construction_path():
    """mini-sglang-style loading: meta construction -> per-parameter
    materialization -> tie_weights re-binding.
    This is the path the Engine actually takes (covered end-to-end by
    tests/test_greedy_decode.py); here we verify the materialization and
    tying invariants in isolation."""
    from python.utils import torch_dtype

    config = Qwen3Config.from_pretrained(MODEL_PATH)
    with torch.device("meta"), torch_dtype(torch.float32):
        model = Qwen3ForCausalLM(config)
    loaded = load_weights(model, MODEL_PATH)
    assert set(model.state_dict()) == loaded
    # No meta parameters left; dtype is decided by the construction context
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # After materialization the tied parameters are independent storage;
    # tie_weights() re-shares them
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
    )


@needs_model
def test_logits_match_hf(our_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).eval()

    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    positions = torch.arange(input_ids.shape[1])

    # Single-request prefill against a freshly allocated pool: out_loc is
    # just the first seq_len slots, and cu_seqlens_q == cu_seqlens_k.
    from python.engine.metadata import AttnMetadata
    from python.kvcache import MHAKVCache

    cfg = our_model.config
    seq_len = input_ids.shape[1]
    kv_cache = MHAKVCache(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_slots=seq_len,
        dtype=next(our_model.parameters()).dtype,
    )
    metadata = AttnMetadata(
        cu_seqlens_q=torch.tensor([0, seq_len]),
        cu_seqlens_k=torch.tensor([0, seq_len]),
        cache_seqlens=torch.tensor([seq_len]),
        max_seqlen_k=seq_len,
        page_table=torch.arange(seq_len).unsqueeze(0),
    )

    with torch.no_grad():
        hidden = our_model(
            input_ids[0], positions, kv_cache, torch.arange(seq_len), metadata
        )
        ours = our_model.compute_logits(hidden)
        ref = hf_model(input_ids).logits

    assert ours.shape == (seq_len, ref.shape[-1])
    torch.testing.assert_close(ours, ref[0], rtol=1e-3, atol=1e-3)
    # same greedy next token
    assert ours[-1].argmax().item() == ref[0, -1].argmax().item()
