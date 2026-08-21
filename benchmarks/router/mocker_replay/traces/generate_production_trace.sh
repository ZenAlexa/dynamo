#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/traces/generate_swe_mooncake.py" \
    --problems 192 \
    --rollouts-per-problem 32 \
    --tool-turns 5 \
    --initial-input-tokens 3750 \
    --assistant-output-tokens 150 \
    --tool-output-tokens 550 \
    --tool-wait-ms 100 \
    --final-output-tokens 256 \
    --trace-block-size 512 \
    --jitter 0.1 \
    --seed 1 \
    --output "$ROOT/traces/generated/swe-production-6144x6.jsonl"
