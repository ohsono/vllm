# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in per-request KV footprint log (VLLM_LOG_REQUEST_KV=1).

Tested on stubs: the two scheduler methods only read the KV cache config, the
coordinator's per-group blocks and a few request fields.
"""

import json
import os
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched import scheduler as sch


class FullAttentionSpec:  # names only: the log reports type(spec).__name__
    block_size = 2096


class MambaSpec:
    block_size = 2096


class _Log:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warnings: list[str] = []

    def info(self, fmt: str, *args: object) -> None:
        self.info_lines.append(fmt % args)

    def warning(self, fmt: str, *args: object) -> None:
        self.warnings.append(fmt % args)


def _stub(freed: list[str], defer: bool = False) -> SimpleNamespace:
    # A null placeholder in a group's block list is not an allocated block.
    full = [SimpleNamespace(is_null=False)] * 3 + [SimpleNamespace(is_null=True)]
    mamba = [SimpleNamespace(is_null=False)]
    held = {"chatcmpl-abc": (full, mamba)}
    stub = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            # Two groups' tensors over one 2000-byte backing allocation: each
            # carries the shared size, so the pool is 2000, not 4000. The
            # host-resident (HiSparse) tensor is a separate pool and is skipped.
            kv_cache_tensors=[
                SimpleNamespace(size=2000),
                SimpleNamespace(size=2000),
                SimpleNamespace(size=9999, host_resident=True),
            ],
            num_blocks=10,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=FullAttentionSpec()),
                SimpleNamespace(kv_cache_spec=MambaSpec()),
            ],
        ),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(get_blocks=lambda rid: held.get(rid, ([], []))),
            free=lambda req: freed.append(req.request_id),
            # The deferred path takes the blocks out of the request's tables.
            pop_blocks_for_free=lambda req: [
                b for blocks in held.pop(req.request_id) for b in blocks
            ],
        ),
        defer_block_free=defer,
        processed_step_seq=0,
        sched_step_seq=7,
        deferred_frees=[],
    )
    stub._log_request_kv = lambda req: sch.Scheduler._log_request_kv(stub, req)
    stub._request_blocks_can_be_freed = (
        lambda req: sch.Scheduler._request_blocks_can_be_freed(stub, req)
    )
    return stub


def _req(finished: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="chatcmpl-abc",
        num_prompt_tokens=900,
        num_output_tokens=40,
        num_preemptions=0,
        last_sched_seq=1,
        is_finished=lambda: finished,
    )


REQ = _req()


def _record(log: _Log) -> dict:
    (line,) = log.info_lines
    tag, payload = line.split(" ", 1)
    assert tag == "COAPKV_REQUEST_KV"
    return json.loads(payload)


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> _Log:
    captured = _Log()
    monkeypatch.setattr(sch, "logger", captured)
    return captured


def test_the_flag_follows_the_env_var_and_defaults_off() -> None:
    expected = os.environ.get("VLLM_LOG_REQUEST_KV", "0") == "1"
    assert sch._LOG_REQUEST_KV is expected


def test_logs_blocks_per_group_and_pool_bytes(log: _Log) -> None:
    sch.Scheduler._log_request_kv(_stub([]), REQ)
    rec = _record(log)
    assert rec["request_id"] == "chatcmpl-abc" and rec["event"] == "finish"
    assert rec["prompt_tokens"] == 900 and rec["output_tokens"] == 40
    assert rec["groups"] == [
        {"kind": "FullAttentionSpec", "block_size": 2096, "blocks": 3},
        {"kind": "MambaSpec", "block_size": 2096, "blocks": 1},
    ]
    assert rec["blocks"] == 4 and rec["page_bytes"] == 200 and rec["bytes"] == 800


def test_a_preemption_line_says_so(log: _Log) -> None:
    # A preempted request is still running when its blocks are freed.
    sch.Scheduler._log_request_kv(_stub([]), _req(finished=False))
    assert _record(log)["event"] == "preempt"


def test_free_logs_only_when_enabled(
    log: _Log, monkeypatch: pytest.MonkeyPatch
) -> None:
    freed: list[str] = []
    stub = _stub(freed)
    monkeypatch.setattr(sch, "_LOG_REQUEST_KV", False)
    sch.Scheduler._free_request_blocks(stub, REQ)
    assert freed == ["chatcmpl-abc"] and log.info_lines == []
    monkeypatch.setattr(sch, "_LOG_REQUEST_KV", True)
    sch.Scheduler._free_request_blocks(stub, REQ)
    assert freed == ["chatcmpl-abc"] * 2 and len(log.info_lines) == 1


def test_a_deferred_free_logs_the_blocks_before_they_are_popped(
    log: _Log, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An in-flight step (last_sched_seq 1 > processed 0) defers the free, and
    # popping empties the request's tables: logging after it would report 0.
    stub = _stub([], defer=True)
    monkeypatch.setattr(sch, "_LOG_REQUEST_KV", True)
    sch.Scheduler._free_request_blocks(stub, REQ)
    assert _record(log)["blocks"] == 4
    assert [seq for seq, _ in stub.deferred_frees] == [7]


def test_a_failure_to_log_never_breaks_the_free(
    log: _Log, monkeypatch: pytest.MonkeyPatch
) -> None:
    freed: list[str] = []
    stub = _stub(freed)
    stub.kv_cache_manager.coordinator = SimpleNamespace(get_blocks=lambda rid: 1 / 0)
    monkeypatch.setattr(sch, "_LOG_REQUEST_KV", True)
    sch.Scheduler._free_request_blocks(stub, REQ)
    assert freed == ["chatcmpl-abc"] and log.info_lines == []
    assert "COAPKV_REQUEST_KV failed" in log.warnings[0]


def test_device_tensors_of_different_sizes_are_refused(log: _Log) -> None:
    # The pool is one shared allocation; two sizes means that no longer holds.
    stub = _stub([])
    stub.kv_cache_config.kv_cache_tensors[1].size = 3000
    sch.Scheduler._log_request_kv(stub, REQ)
    assert log.info_lines == [] and "2 sizes" in log.warnings[0]


def test_blocks_for_more_groups_than_configured_are_refused(log: _Log) -> None:
    # Pairing groups with counts must not silently drop or mislabel a group.
    stub = _stub([])
    stub.kv_cache_manager.coordinator = SimpleNamespace(
        get_blocks=lambda rid: ([], [], [])
    )
    sch.Scheduler._log_request_kv(stub, REQ)
    assert log.info_lines == [] and "COAPKV_REQUEST_KV failed" in log.warnings[0]
