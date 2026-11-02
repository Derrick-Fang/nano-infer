"""Pure-logic tests for core.py (no model weights needed)."""

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
    assert req.extend_len == 3  # prefill computes the whole prompt
    assert req.remain_len == 5
    assert req.can_decode


def test_empty_prompt_rejected():
    with pytest.raises(AssertionError):
        make_req(prompt=())


def test_complete_one_and_append():
    # mini-sglang's paired usage: complete_one() after the forward, then
    # append_host() with the sampled token
    req = make_req()
    req.complete_one()
    req.append_host(99)
    assert req.cached_len == 3
    assert req.device_len == 4
    assert req.input_ids == [10, 20, 30, 99]
    assert req.remain_len == 4


def test_extend_len_decode_step():
    req = make_req()
    req.complete_one()  # prefill done: cached catches up with device
    req.append_host(99)
    # each decode forward computes only 1 new token (the one complete_one
    # accounted for in advance)
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
    # DecodeManager.running_reqs is a set[Req] and must hash by identity
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
