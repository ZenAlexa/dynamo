#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate deterministic session shards from run_generate.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def sum_bins(results: list[dict[str, Any]], key: str) -> list[int]:
    size = max((len(result[key]) for result in results), default=0)
    return [
        sum(
            result[key][second] if second < len(result[key]) else 0
            for result in results
        )
        for second in range(size)
    ]


def weighted_mean(results: list[dict[str, Any]], key: str) -> float:
    weighted = [
        (float(result[key]), int(result["successful_requests"]))
        for result in results
        if math.isfinite(float(result[key])) and int(result["successful_requests"])
    ]
    total = sum(weight for _, weight in weighted)
    return (
        sum(value * weight for value, weight in weighted) / total if total else math.nan
    )


def aggregate(paths: list[Path]) -> dict[str, Any]:
    results = [json.loads(path.read_text()) for path in paths]
    if not results:
        raise ValueError("at least one shard result is required")
    shard_counts = {int(result["session_shard_count"]) for result in results}
    shard_indices = [int(result["session_shard_index"]) for result in results]
    if shard_counts != {len(results)} or sorted(shard_indices) != list(
        range(len(results))
    ):
        raise ValueError(
            f"expected exactly one result for each of {len(results)} shards; "
            f"counts={sorted(shard_counts)}, indices={sorted(shard_indices)}"
        )
    monitored = [result for result in results if result.get("metric_samples")]
    if len(monitored) != 1:
        raise ValueError(f"expected exactly one metrics shard, got {len(monitored)}")
    metrics = monitored[0]
    wall_s = max(float(result["wall_s"]) for result in results)
    output_tokens = sum(int(result["output_tokens"]) for result in results)
    successful_requests = sum(int(result["successful_requests"]) for result in results)
    url_requests: dict[str, int] = {}
    for result in results:
        for url, count in result["url_requests"].items():
            url_requests[url] = url_requests.get(url, 0) + int(count)
    errors = [error for result in results for error in result["errors"]]
    return {
        "client_shards": len(results),
        "source_results": [str(path) for path in paths],
        "urls": sorted(url_requests),
        "url_requests": url_requests,
        "sessions": sum(int(result["sessions"]) for result in results),
        "expected_requests": sum(
            int(result["expected_requests"]) for result in results
        ),
        "successful_requests": successful_requests,
        "failed_sessions": sum(int(result["failed_sessions"]) for result in results),
        "concurrency": sum(int(result["concurrency"]) for result in results),
        "delay_scale": float(results[0]["delay_scale"]),
        "wall_s": wall_s,
        "request_rate": successful_requests / wall_s if wall_s else math.nan,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall_s if wall_s else math.nan,
        "latency_mean_s": weighted_mean(results, "latency_mean_s"),
        "latency_p95_client_range_s": [
            min(float(result["latency_p95_s"]) for result in results),
            max(float(result["latency_p95_s"]) for result in results),
        ],
        "ttft_mean_s": weighted_mean(results, "ttft_mean_s"),
        "ttft_p95_client_range_s": [
            min(float(result["ttft_p95_s"]) for result in results),
            max(float(result["ttft_p95_s"]) for result in results),
        ],
        "sglang_queue_mean": metrics["sglang_queue_mean"],
        "sglang_queue_peak": metrics["sglang_queue_peak"],
        "router_queue_mean": metrics["router_queue_mean"],
        "router_queue_peak": metrics["router_queue_peak"],
        "cache_hit_mean": metrics["cache_hit_mean"],
        "cache_hit_final": metrics["cache_hit_final"],
        "transport_errors_final": metrics["transport_errors_final"],
        "metric_samples": metrics["metric_samples"],
        "per_second_output_tokens": sum_bins(results, "per_second_output_tokens"),
        "per_second_requests": sum_bins(results, "per_second_requests"),
        "errors": errors[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
