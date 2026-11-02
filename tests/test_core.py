"""core.py 的纯逻辑测试（不依赖模型权重）。"""

import pytest

from python.core import Batch, Req, SamplingParams


def make_req(prompt=(10, 20, 30), max_tokens=5, **kwargs) -> Req:
    kwargs.setdefault("uid", 0)
    return Req(
        input_ids=list(prompt),
        cached_len=0,
        output_len=max_tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        **kwargs,
    )


def test_initial_accounting():
    req = make_req()
    assert req.device_len == 3
    assert req.max_device_len == 3 + 5
    assert req.cached_len == 0
    assert req.extend_len == 3  # prefill 要算全部 prompt
    assert req.remain_len == 5
    assert req.can_decode


def test_empty_prompt_rejected():
    with pytest.raises(AssertionError):
        make_req(prompt=())


def test_complete_one_and_append():
    # mini-sglang 的配对用法：forward 后 complete_one()，采样结果 append_host()
    req = make_req()
    req.complete_one()
    req.append_host(99)
    assert req.cached_len == 3
    assert req.device_len == 4
    assert req.input_ids == [10, 20, 30, 99]
    assert req.remain_len == 4


def test_extend_len_decode_step():
    req = make_req()
    req.complete_one()  # prefill 完成：cached 追上 device
    req.append_host(99)
    # decode 每步前向只算 1 个新 token（complete_one 预支的那个）
    assert req.extend_len == 1


def test_can_decode_until_max_tokens():
    req = make_req(max_tokens=2)
    req.complete_one()
    req.append_host(1)
    assert req.can_decode
    req.complete_one()
    req.append_host(2)
    assert not req.can_decode  # remain_len == 0
    assert req.device_len == req.max_device_len


def test_eq_false_identity_hash():
    # DecodeManager.running_reqs 是 set[Req]，必须按身份哈希
    a, b = make_req(uid=0), make_req(uid=0)
    assert a != b
    assert len({a, b}) == 2


def test_is_greedy():
    assert SamplingParams(temperature=0.0).is_greedy
    assert SamplingParams(temperature=0.7, top_k=1).is_greedy
    assert not SamplingParams(temperature=0.7).is_greedy


def test_repr():
    assert "uid=0" in repr(make_req())


def test_batch_phase():
    reqs = [make_req(uid=0), make_req(uid=1)]
    prefill = Batch(reqs=reqs, phase="prefill")
    assert prefill.is_prefill and not prefill.is_decode
    assert prefill.size == 2
    decode = Batch(reqs=reqs, phase="decode")
    assert decode.is_decode and not decode.is_prefill
