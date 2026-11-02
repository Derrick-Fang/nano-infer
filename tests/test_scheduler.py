"""Pure-logic tests for the Scheduler's Managers (no model loading).

Covers R3's admission and batching rules:
- PrefillManager: FIFO admission, max_running_req slots, token budget,
  whole-batch admission of an over-budget single prompt;
- DecodeManager: filter_reqs merging and dropping, uid-sorted batching.
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
        # Once a PendingReq becomes a Req: no prefix cache, so cached_len
        # is always 0
        assert all(req.cached_len == 0 for req in batch.reqs)
        assert all(req.extend_len == 4 for req in batch.reqs)

    def test_max_running_req_limits_admission(self):
        decode_manager = DecodeManager()
        decode_manager.running_reqs.add(make_running_req(uid=99))  # one already running
        manager = PrefillManager(max_running_req=2, decode_manager=decode_manager)
        for uid in range(3):
            manager.add_one_req(make_pending(uid, input_len=2))
        batch = manager.schedule_next_batch(prefill_budget=1024)
        # 1 of the 2 slots is taken by the running request; only 1 admitted
        assert [req.uid for req in batch.reqs] == [0]
        assert [r.uid for r in manager.pending_list] == [1, 2]

    def test_no_slot_returns_none(self):
        decode_manager = DecodeManager()
        decode_manager.running_reqs.add(make_running_req(uid=99))
        manager = PrefillManager(max_running_req=1, decode_manager=decode_manager)
        manager.add_one_req(make_pending(uid=0, input_len=2))
        assert manager.schedule_next_batch(prefill_budget=1024) is None
        assert len(manager.pending_list) == 1  # stays queued for the next round

    def test_token_budget_splits_batches(self):
        manager = PrefillManager(max_running_req=8, decode_manager=DecodeManager())
        for uid in range(3):
            manager.add_one_req(make_pending(uid, input_len=4))
        batch = manager.schedule_next_batch(prefill_budget=10)  # only two fit
        assert [req.uid for req in batch.reqs] == [0, 1]
        assert [r.uid for r in manager.pending_list] == [2]

    def test_oversized_prompt_admitted_whole_when_first(self):
        """R3 has no ChunkedReq: a single prompt exceeding the whole budget
        is admitted whole when it is the first of the batch (otherwise it
        could never be scheduled)."""
        manager = PrefillManager(max_running_req=8, decode_manager=DecodeManager())
        manager.add_one_req(make_pending(uid=0, input_len=10))
        manager.add_one_req(make_pending(uid=1, input_len=1))
        batch = manager.schedule_next_batch(prefill_budget=4)
        assert [req.uid for req in batch.reqs] == [0]  # the oversized one gets its own batch
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
        done.complete_one()  # device_len reaches max_device_len -> can_decode False
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
        manager.remove_req(req)  # discard semantics: repeated removal is fine
