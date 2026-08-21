#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a fixed-shape, closed-loop SWE rollout Mooncake trace."""

import argparse
import json
import math
import random
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def sample(rng: random.Random, value: int | float, jitter: float, minimum: int) -> int:
    if jitter == 0:
        return round(value)
    low = max(minimum, round(value * (1 - jitter)))
    high = max(low, round(value * (1 + jitter)))
    return rng.randint(low, high)


def sample_delay_ms(rng: random.Random, value: float, jitter: float) -> float:
    if jitter == 0:
        return value
    return rng.uniform(max(0.0, value * (1 - jitter)), value * (1 + jitter))


def extend_hash_ids(hash_ids: list[int], input_length: int, block_size: int, next_hash: int) -> int:
    required_blocks = math.ceil(input_length / block_size)
    while len(hash_ids) < required_blocks:
        hash_ids.append(next_hash)
        next_hash += 1
    return next_hash


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    next_hash = 1_000_000_000

    for problem_index in range(args.problems):
        initial_input = sample(rng, args.initial_input_tokens, args.jitter, 1)
        shared_hash_base = (problem_index + 1) * 1_000_000
        shared_blocks = math.ceil(initial_input / args.trace_block_size)
        initial_hashes = [shared_hash_base + block for block in range(shared_blocks)]
        strict_priority = args.problems - problem_index

        for rollout_index in range(args.rollouts_per_problem):
            session_id = f"problem-{problem_index}:rollout-{rollout_index}"
            input_length = initial_input
            hash_ids = list(initial_hashes)

            for turn_index in range(args.tool_turns):
                assistant_output = sample(rng, args.assistant_output_tokens, args.jitter, 1)
                tool_output = sample(rng, args.tool_output_tokens, args.jitter, 1)
                tool_wait_ms = sample_delay_ms(rng, args.tool_wait_ms, args.jitter)
                row: dict[str, object] = {
                    "session_id": session_id,
                    "input_length": input_length,
                    "output_length": assistant_output,
                    "hash_ids": list(hash_ids),
                    "extra": {
                        "nvext": {"agent_hints": {"strict_priority": strict_priority}}
                    },
                }
                if turn_index == 0:
                    row["timestamp"] = 0
                else:
                    row["delay"] = prior_tool_wait_ms
                rows.append(row)

                input_length = tool_output
                hash_ids = []
                next_hash = extend_hash_ids(
                    hash_ids, input_length, args.trace_block_size, next_hash
                )
                prior_tool_wait_ms = tool_wait_ms

            rows.append(
                {
                    "session_id": session_id,
                    "delay": prior_tool_wait_ms,
                    "input_length": input_length,
                    "output_length": sample(rng, args.final_output_tokens, args.jitter, 1),
                    "hash_ids": list(hash_ids),
                    "extra": {
                        "nvext": {"agent_hints": {"strict_priority": strict_priority}}
                    },
                }
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=positive_int, required=True)
    parser.add_argument("--rollouts-per-problem", type=positive_int, required=True)
    parser.add_argument("--tool-turns", type=positive_int, required=True)
    parser.add_argument("--initial-input-tokens", type=positive_int, required=True)
    parser.add_argument("--assistant-output-tokens", type=positive_int, required=True)
    parser.add_argument("--tool-output-tokens", type=positive_int, required=True)
    parser.add_argument("--tool-wait-ms", type=nonnegative_float, required=True)
    parser.add_argument("--final-output-tokens", type=positive_int, required=True)
    parser.add_argument("--trace-block-size", type=positive_int, default=512)
    parser.add_argument("--jitter", type=nonnegative_float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.jitter > 1:
        parser.error("--jitter must be at most 1")
    return args


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
