#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redis-free, closed-loop native SGLang /generate replay for the MAI rig."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp


MASK64 = (1 << 64) - 1
METRIC_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def token_for(hash_id: int, offset: int) -> int:
    """Map a trace block and offset to a stable model-vocabulary token."""
    value = (hash_id ^ (offset * 0x9E3779B97F4A7C15)) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return 100 + (value % 100_000)


def materialize_input(row: dict[str, Any], block_size: int) -> list[int]:
    input_length = int(row["input_length"])
    hash_ids = row.get("hash_ids") or []
    required_blocks = math.ceil(input_length / block_size)
    if len(hash_ids) != required_blocks:
        raise ValueError(
            f"{row['session_id']}: expected {required_blocks} hash blocks, "
            f"got {len(hash_ids)}"
        )
    tokens: list[int] = []
    remaining = input_length
    for hash_id in hash_ids:
        count = min(block_size, remaining)
        tokens.extend(token_for(int(hash_id), offset) for offset in range(count))
        remaining -= count
    return tokens


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def load_sessions(
    trace_path: Path, max_sessions: int | None, max_turns: int | None
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open(encoding="utf-8") as trace:
        for line in trace:
            row = json.loads(line)
            grouped[str(row["session_id"])].append(row)
    sessions = list(grouped.items())
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    if max_turns is not None:
        sessions = [(session_id, rows[:max_turns]) for session_id, rows in sessions]
    return sessions


def ports(path: Path | None) -> list[int]:
    if path is None:
        return []
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def metric_values(text: str, name: str) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        match = METRIC_RE.match(line)
        if match and match.group(1) == name:
            values.append(float(match.group(2)))
    return values


async def fetch_metrics(
    client: aiohttp.ClientSession, metric_ports: list[int]
) -> list[str]:
    async def fetch(port: int) -> str:
        async with client.get(f"http://127.0.0.1:{port}/metrics") as response:
            response.raise_for_status()
            return await response.text()

    results = await asyncio.gather(
        *(fetch(port) for port in metric_ports), return_exceptions=True
    )
    return [result for result in results if isinstance(result, str)]


async def sample_metrics(
    client: aiohttp.ClientSession,
    frontend_ports: list[int],
    mocker_ports: list[int],
    elapsed_s: float,
) -> dict[str, float]:
    frontend, mocker = await asyncio.gather(
        fetch_metrics(client, frontend_ports), fetch_metrics(client, mocker_ports)
    )
    router_queue = sum(
        sum(metric_values(body, "dynamo_frontend_router_queue_pending_requests"))
        for body in frontend
    )
    transport_errors = sum(
        sum(metric_values(body, "dynamo_transport_tcp_errors_total"))
        for body in frontend
    )
    sglang_queue = sum(
        sum(metric_values(body, "sglang:num_queue_reqs")) for body in mocker
    )
    cache_values = [
        value
        for body in mocker
        for value in metric_values(body, "sglang:cache_hit_rate")
    ]
    return {
        "elapsed_s": elapsed_s,
        "router_queue": router_queue,
        "sglang_queue": sglang_queue,
        "cache_hit": statistics.fmean(cache_values) if cache_values else math.nan,
        "transport_errors": transport_errors,
        "frontend_scrapes": float(len(frontend)),
        "mocker_scrapes": float(len(mocker)),
    }


async def generate(
    client: aiohttp.ClientSession,
    url: str,
    session_id: str,
    turn: int,
    input_ids: list[int],
    row: dict[str, Any],
    run_started: float,
    token_bins: dict[int, int],
) -> tuple[list[int], dict[str, float]]:
    expected_tokens = int(row["output_length"])
    priority = int(row["extra"]["nvext"]["agent_hints"]["strict_priority"])
    payload = {
        "rid": f"{session_id}:{turn}",
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": expected_tokens,
            "ignore_eos": True,
        },
        "priority": priority,
        "stream": True,
    }
    headers = {"X-Dynamo-Session-ID": session_id}
    started = time.perf_counter()
    first_token_at: float | None = None
    output_ids: list[int] = []
    terminal_count = 0
    saw_done = False

    async with client.post(
        f"{url.rstrip('/')}/generate", json=payload, headers=headers
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"stream error: {event['error']}")
            chunk_ids = event.get("output_ids") or []
            if not isinstance(chunk_ids, list) or not all(
                isinstance(token, int) for token in chunk_ids
            ):
                raise RuntimeError(f"invalid output_ids: {chunk_ids!r}")
            if chunk_ids:
                observed_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = observed_at
                token_bins[int(observed_at - run_started)] += len(chunk_ids)
            output_ids.extend(chunk_ids)
            meta = event.get("meta_info")
            if not isinstance(meta, dict):
                raise RuntimeError(f"missing meta_info: {event!r}")
            completion_tokens = meta.get("completion_tokens")
            if completion_tokens != len(output_ids):
                raise RuntimeError(
                    f"completion_tokens={completion_tokens}, observed={len(output_ids)}"
                )
            if meta.get("finish_reason") is not None:
                terminal_count += 1

    finished = time.perf_counter()
    if not saw_done:
        raise RuntimeError("stream ended without [DONE]")
    if terminal_count != 1:
        raise RuntimeError(f"expected one terminal response, got {terminal_count}")
    if len(output_ids) != expected_tokens:
        raise RuntimeError(
            f"expected {expected_tokens} output IDs, received {len(output_ids)}"
        )
    if first_token_at is None:
        raise RuntimeError("stream produced no output token")
    return output_ids, {
        "latency_s": finished - started,
        "ttft_s": first_token_at - started,
        "output_tokens": float(len(output_ids)),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    sessions = load_sessions(args.trace, args.max_sessions, args.max_turns)
    if not 0 <= args.session_shard_index < args.session_shard_count:
        raise ValueError(
            "session shard index must be in "
            f"[0, {args.session_shard_count}), got {args.session_shard_index}"
        )
    sessions = sessions[args.session_shard_index :: args.session_shard_count]
    urls = args.url or ["http://127.0.0.1:28001"]
    frontend_ports = ports(args.frontend_ports_file)
    mocker_ports = ports(args.mocker_ports_file)
    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency * 2, 64),
        limit_per_host=0,
        keepalive_timeout=30,
    )
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300)
    semaphore = asyncio.Semaphore(args.concurrency)
    request_stats: list[dict[str, float]] = []
    error_samples: list[str] = []
    error_count = 0
    token_bins: dict[int, int] = defaultdict(int)
    request_bins: dict[int, int] = defaultdict(int)
    url_requests: dict[str, int] = defaultdict(int)
    metric_samples: list[dict[str, float]] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as client:
        run_started = time.perf_counter()
        stop_metrics = asyncio.Event()

        async def metrics_loop() -> None:
            if not frontend_ports and not mocker_ports:
                return
            metrics_timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=metrics_timeout) as metrics_client:
                while not stop_metrics.is_set():
                    try:
                        metric_samples.append(
                            await sample_metrics(
                                metrics_client,
                                frontend_ports,
                                mocker_ports,
                                time.perf_counter() - run_started,
                            )
                        )
                    except Exception as error:
                        metric_samples.append(
                            {
                                "elapsed_s": time.perf_counter() - run_started,
                                "sample_error": str(error),
                            }
                        )
                    try:
                        await asyncio.wait_for(
                            stop_metrics.wait(), timeout=args.metrics_interval
                        )
                    except TimeoutError:
                        pass

        metrics_task = asyncio.create_task(metrics_loop())

        async def run_session(
            session_index: int, session_id: str, rows: list[dict[str, Any]]
        ) -> None:
            nonlocal error_count
            url = urls[session_index % len(urls)]
            async with semaphore:
                context: list[int] = []
                try:
                    for turn, row in enumerate(rows):
                        if turn:
                            await asyncio.sleep(
                                float(row.get("delay", 0.0)) * args.delay_scale / 1000.0
                            )
                        new_input = materialize_input(row, args.trace_block_size)
                        if turn == 0:
                            context = new_input
                        else:
                            context.extend(new_input)
                        output_ids, stats = await generate(
                            client,
                            url,
                            session_id,
                            turn,
                            context,
                            row,
                            run_started,
                            token_bins,
                        )
                        request_stats.append(stats)
                        url_requests[url] += 1
                        request_bins[int(time.perf_counter() - run_started)] += 1
                        context.extend(output_ids)
                except Exception as error:
                    error_count += 1
                    if len(error_samples) < 50:
                        error_samples.append(
                            f"{session_id}: {type(error).__name__}: {error}"
                        )

        try:
            await asyncio.gather(
                *(
                    run_session(index, session_id, rows)
                    for index, (session_id, rows) in enumerate(sessions)
                )
            )
        finally:
            stop_metrics.set()
            await metrics_task
        wall_s = time.perf_counter() - run_started

    latencies = [entry["latency_s"] for entry in request_stats]
    ttfts = [entry["ttft_s"] for entry in request_stats]
    output_tokens = int(sum(entry["output_tokens"] for entry in request_stats))
    expected_requests = sum(len(rows) for _, rows in sessions)
    observed_samples = [
        sample for sample in metric_samples if "sample_error" not in sample
    ]
    cache_samples = [
        sample["cache_hit"]
        for sample in observed_samples
        if math.isfinite(sample.get("cache_hit", math.nan))
    ]
    max_second = max(token_bins.keys() | request_bins.keys(), default=-1)
    return {
        "urls": urls,
        "url_requests": dict(url_requests),
        "sessions": len(sessions),
        "session_shard_index": args.session_shard_index,
        "session_shard_count": args.session_shard_count,
        "expected_requests": expected_requests,
        "successful_requests": len(request_stats),
        "failed_sessions": error_count,
        "concurrency": args.concurrency,
        "delay_scale": args.delay_scale,
        "wall_s": wall_s,
        "request_rate": len(request_stats) / wall_s if wall_s else math.nan,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall_s if wall_s else math.nan,
        "latency_mean_s": statistics.fmean(latencies) if latencies else math.nan,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else math.nan,
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "sglang_queue_mean": statistics.fmean(
            sample["sglang_queue"] for sample in observed_samples
        )
        if observed_samples
        else math.nan,
        "sglang_queue_peak": max(
            (sample["sglang_queue"] for sample in observed_samples), default=math.nan
        ),
        "router_queue_mean": statistics.fmean(
            sample["router_queue"] for sample in observed_samples
        )
        if observed_samples
        else math.nan,
        "router_queue_peak": max(
            (sample["router_queue"] for sample in observed_samples), default=math.nan
        ),
        "cache_hit_mean": statistics.fmean(cache_samples)
        if cache_samples
        else math.nan,
        "cache_hit_final": cache_samples[-1] if cache_samples else math.nan,
        "transport_errors_final": observed_samples[-1]["transport_errors"]
        if observed_samples
        else math.nan,
        "metric_samples": metric_samples,
        "per_second_output_tokens": [
            token_bins[second] for second in range(max_second + 1)
        ],
        "per_second_requests": [
            request_bins[second] for second in range(max_second + 1)
        ],
        "errors": error_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--session-shard-index", type=int, default=0)
    parser.add_argument("--session-shard-count", type=int, default=1)
    parser.add_argument("--trace-block-size", type=int, default=512)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--frontend-ports-file", type=Path)
    parser.add_argument("--mocker-ports-file", type=Path)
    parser.add_argument("--metrics-interval", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2, sort_keys=True))
