# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for native SGLang streaming through stock Mocker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import ServicePorts

TEST_MODEL = QWEN
INPUT_IDS = [11, 12, 13]
OUTPUT_IDS = [101, 202, 303]
REQUEST_ID = "native-replay"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.sglang,
    pytest.mark.model(TEST_MODEL),
]


@pytest.fixture(scope="function")
def sglang_generate_mocker(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_tokenizers: object,
    tmp_path: Path,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    replay_trace = tmp_path / "sglang-response-replay.jsonl"
    replay_trace.write_text(
        json.dumps(
            {
                "request_id": REQUEST_ID,
                "output_length": len(OUTPUT_IDS),
                "output_token_ids": OUTPUT_IDS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={"DYN_SGLANG_ENABLE_GENERATE": "1"},
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            system_port,
            extra_args=[
                "--engine-type",
                "sglang",
                "--sglang-generate",
                "--response-replay-trace-path",
                str(replay_trace),
            ],
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port,
                model=TEST_MODEL,
            )
            yield frontend_port


def _native_events(frontend_port: int) -> tuple[list[dict[str, Any]], bool]:
    with requests.post(
        f"http://localhost:{frontend_port}/generate",
        json={
            "rid": REQUEST_ID,
            "input_ids": INPUT_IDS,
            "sampling_params": {
                "max_new_tokens": len(OUTPUT_IDS),
                "ignore_eos": True,
                "n": 1,
            },
            "stream": True,
            "return_logprob": True,
            "top_logprobs_num": 2,
            "logprob_start_len": 1,
        },
        stream=True,
        timeout=60,
    ) as response:
        assert response.status_code == 200, response.text
        events: list[dict[str, Any]] = []
        done = False
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            assert isinstance(event, dict), event
            assert "error" not in event, event
            events.append(event)
    return events, done


@pytest.mark.timeout(120)
def test_native_generate_replays_incremental_ids_and_keeps_openai_canonical(
    sglang_generate_mocker: int,
) -> None:
    frontend_port = sglang_generate_mocker
    events, done = _native_events(frontend_port)

    assert done
    assert events
    terminal_events = [
        event
        for event in events
        if event.get("meta_info", {}).get("finish_reason") is not None
    ]
    assert terminal_events == [events[-1]]

    replayed: list[int] = []
    for event in events:
        output_ids = event.get("output_ids")
        meta_info = event.get("meta_info")
        assert isinstance(output_ids, list), event
        assert isinstance(meta_info, dict), event
        replayed.extend(output_ids)
        assert meta_info.get("id") == REQUEST_ID, event
        assert meta_info.get("prompt_tokens") == len(INPUT_IDS), event
        assert meta_info.get("completion_tokens") == len(replayed), event

        output_logprobs = meta_info.get("output_token_logprobs")
        output_top_logprobs = meta_info.get("output_top_logprobs")
        assert isinstance(output_logprobs, list), event
        assert len(output_logprobs) == len(output_ids), event
        assert isinstance(output_top_logprobs, list), event
        assert len(output_top_logprobs) == len(output_ids), event
        for token_id, selected, top in zip(
            output_ids,
            output_logprobs,
            output_top_logprobs,
            strict=True,
        ):
            assert selected[1] == token_id, event
            assert isinstance(top, list) and len(top) == 2, event

    assert replayed == OUTPUT_IDS
    terminal_meta = events[-1]["meta_info"]
    assert terminal_meta["finish_reason"] == {"type": "length"}
    assert terminal_meta["completion_tokens"] == len(OUTPUT_IDS)
    assert terminal_meta["input_token_logprobs"][0] == [None, INPUT_IDS[1], None]
    assert terminal_meta["input_top_logprobs"][0] is None

    # OpenAI requests carry no sglang_tito payload, so they skip the adapter.
    completion = requests.post(
        f"http://localhost:{frontend_port}/v1/completions",
        json={"model": TEST_MODEL, "prompt": "ping", "max_tokens": 2},
        timeout=60,
    )
    assert completion.status_code == 200, completion.text
    completion_body = completion.json()
    assert completion_body.get("object") == "text_completion"
    assert completion_body.get("choices")
    assert "sglang_response" not in completion.text
