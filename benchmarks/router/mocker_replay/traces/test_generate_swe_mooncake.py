# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("generate_swe_mooncake.py")
SPEC = importlib.util.spec_from_file_location("generate_swe_mooncake", MODULE)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def args(**overrides):
    values = {
        "problems": 2,
        "rollouts_per_problem": 3,
        "tool_turns": 2,
        "initial_input_tokens": 100,
        "assistant_output_tokens": 10,
        "tool_output_tokens": 30,
        "tool_wait_ms": 200,
        "final_output_tokens": 12,
        "trace_block_size": 64,
        "jitter": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class GenerateSweMooncakeTest(unittest.TestCase):
    def test_closed_loop_shape_and_wire_priority(self) -> None:
        rows = GENERATOR.build_rows(args())
        self.assertEqual(len(rows), 18)
        rollout = [row for row in rows if row["session_id"] == "problem-0:rollout-0"]
        self.assertEqual([row["input_length"] for row in rollout], [100, 30, 30])
        self.assertEqual([row.get("delay") for row in rollout], [None, 200, 200])
        self.assertEqual(
            {row["extra"]["nvext"]["agent_hints"]["strict_priority"] for row in rollout},
            {2},
        )

    def test_seeded_generation_is_deterministic(self) -> None:
        self.assertEqual(GENERATOR.build_rows(args()), GENERATOR.build_rows(args()))

    def test_non_finite_delay_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            GENERATOR.nonnegative_float("nan")


if __name__ == "__main__":
    unittest.main()
