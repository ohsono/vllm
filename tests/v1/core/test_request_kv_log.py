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


def _stub(freed: list[str]) -> SimpleNamespace:
    # A null placeholder in a group's block list is not an allocated block.
    full = [SimpleNamespace(is_null=False)] * 3 + [SimpleNamespace(is_null=True)]
    mamba = [SimpleNamespace(is_null=False)]
    stub = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=1000), SimpleNamespace(size=1000)],
            num_blocks=10,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=FullAttentionSpec()),
                SimpleNamespace(kv_cache_spec=MambaSpec()),
            ],
        ),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(get_blocks=lambda rid: (full, mamba)),
            free=lambda req: freed.append(req.request_id),
        ),
        defer_block_free=False,
    )
    stub._log_request_kv = lambda req: sch.Scheduler._log_request_kv(stub, req)
    return stub


REQ = SimpleNamespace(
    request_id="chatcmpl-abc",
    num_prompt_tokens=900,
    num_output_tokens=40,
    num_preemptions=0,
)


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
    (line,) = log.info_lines
    tag, payload = line.split(" ", 1)
    rec = json.loads(payload)
    assert tag == "COAPKV_REQUEST_KV"
    assert rec["request_id"] == "chatcmpl-abc"
    assert rec["prompt_tokens"] == 900 and rec["output_tokens"] == 40
    assert rec["groups"] == [
        {"kind": "FullAttentionSpec", "block_size": 2096, "blocks": 3},
        {"kind": "MambaSpec", "block_size": 2096, "blocks": 1},
    ]
    assert rec["blocks"] == 4 and rec["page_bytes"] == 200 and rec["bytes"] == 800


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


def test_a_failure_to_log_never_breaks_the_free(log: _Log) -> None:
    stub = _stub([])
    stub.kv_cache_manager.coordinator = SimpleNamespace(get_blocks=lambda rid: 1 / 0)
    sch.Scheduler._log_request_kv(stub, REQ)
    assert log.info_lines == []
    assert "COAPKV_REQUEST_KV failed" in log.warnings[0]
