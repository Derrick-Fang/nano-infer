"""Scheduler 各 Manager 的纯逻辑测试（不加载模型）。

覆盖 R3 的准入与组批规则：
- PrefillManager：FIFO 准入、max_running_req 名额、token 预算、
  单条超预算整批放行；
- DecodeManager：filter_reqs 的并入与丢弃、按 uid 排序组批。
"""

from python.core import Req, SamplingParams
from python.scheduler.decode import DecodeManager
from python.scheduler.prefill import PrefillManager
from python.scheduler.utils import PendingReq


def make_pending(uid: int, input_len: int, max_tokens: int = 8) -> PendingReq:
    return PendingReq(
        uid=uid,
        input_ids=list(range(input_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_running_req(uid: int, prompt_len: int = 3, max_tokens: int = 8) -> Req:
    return Req(
        input_ids=list(range(prompt_len)),
        cached_len=0,
        output_len=max_tokens,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class TestPendingReq:
    def test_lengths(self):
        req = make_pending(uid=0, input_len=5, max_tokens=3)
        assert req.input_len == 5
        assert req.output_len == 3


class TestPrefillManager:
    def test_schedule_all_when_capacity_allows(self):
        manager = PrefillManager(max_running_req=4, decode_manager=DecodeManager())
        for uid in range(3):
            manager.add_one_req(make_pending(uid, input_len=4))
        batch = manager.schedule_next_batch(prefill_budget=1024)
        assert batch is not None and batch.is_prefill
        assert [req.uid for req in batch.reqs] == [0, 1, 2]  # FIFO
        assert manager.pending_list == []
        # PendingReq 变成 Req 后：无 prefix cache，cached_len 恒 0
        assert all(req.cached_len == 0 for req in batch.reqs)
        assert all(req.extend_len == 4 for req in batch.reqs)

    def test_max_running_req_limits_admission(self):
        decode_manager = DecodeManager()
        decode_manager.running_reqs.add(make_running_req(uid=99))  # 一个在跑
        manager = PrefillManager(max_running_req=2, decode_manager=decode_manager)
        for uid in range(3):
            manager.add_one_req(make_pending(uid, input_len=2))
        batch = manager.schedule_next_batch(prefill_budget=1024)
        # 2 个名额被在跑请求占掉 1 个，本轮只准进 1 个
        assert [req.uid for req in batch.reqs] == [0]
        assert [r.uid for r in manager.pending_list] == [1, 2]

    def test_no_slot_returns_none(self):
        decode_manager = DecodeManager()
        decode_manager.running_reqs.add(make_running_req(uid=99))
        manager = PrefillManager(max_running_req=1, decode_manager=decode_manager)
        manager.add_one_req(make_pending(uid=0, input_len=2))
        assert manager.schedule_next_batch(prefill_budget=1024) is None
        assert len(manager.pending_list) == 1  # 留在队列里等下一轮

    def test_token_budget_splits_batches(self):
        manager = PrefillManager(max_running_req=8, decode_manager=DecodeManager())
        for uid in range(3):
            manager.add_one_req(make_pending(uid, input_len=4))
        batch = manager.schedule_next_batch(prefill_budget=10)  # 只装得下两条
        assert [req.uid for req in batch.reqs] == [0, 1]
        assert [r.uid for r in manager.pending_list] == [2]

    def test_oversized_prompt_admitted_whole_when_first(self):
        """R3 无 ChunkedReq：单条超过整个预算的 prompt 若是本批第一条，
        整批放行（否则永远无法被调度）。"""
        manager = PrefillManager(max_running_req=8, decode_manager=DecodeManager())
        manager.add_one_req(make_pending(uid=0, input_len=10))
        manager.add_one_req(make_pending(uid=1, input_len=1))
        batch = manager.schedule_next_batch(prefill_budget=4)
        assert [req.uid for req in batch.reqs] == [0]  # 超长独占一批
        assert [r.uid for r in manager.pending_list] == [1]
        batch2 = manager.schedule_next_batch(prefill_budget=4)
        assert [req.uid for req in batch2.reqs] == [1]

    def test_empty_pending_returns_none(self):
        manager = PrefillManager(max_running_req=8, decode_manager=DecodeManager())
        assert manager.schedule_next_batch(prefill_budget=1024) is None
        assert not manager.runnable


class TestDecodeManager:
    def test_filter_reqs_unions_and_drops_undecodable(self):
        manager = DecodeManager()
        keep = make_running_req(uid=0)
        done = make_running_req(uid=1, max_tokens=1)
        done.complete_one()  # device_len 到达 max_device_len -> can_decode False
        manager.filter_reqs([keep, done])
        assert manager.running_reqs == {keep}

    def test_schedule_next_batch_sorted_by_uid(self):
        manager = DecodeManager()
        for uid in [2, 0, 1]:
            manager.filter_reqs([make_running_req(uid=uid)])
        batch = manager.schedule_next_batch()
        assert batch is not None and batch.is_decode
        assert [req.uid for req in batch.reqs] == [0, 1, 2]

    def test_remove_req_and_runnable(self):
        manager = DecodeManager()
        req = make_running_req(uid=0)
        manager.filter_reqs([req])
        assert manager.runnable
        manager.remove_req(req)
        assert not manager.runnable
        assert manager.schedule_next_batch() is None
        manager.remove_req(req)  # discard 语义：重复移除不报错
