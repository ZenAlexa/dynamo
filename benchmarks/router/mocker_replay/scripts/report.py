# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print deltas for public Prometheus counters from an AIPerf artifact directory."""

from __future__ import annotations

import re
import sys
from pathlib import Path

COUNTERS = (
    "dynamo_component_router_requests_total",
    "dynamo_frontend_model_rejection_total",
)


def counters(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in COUNTERS:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))
    return values


def main() -> None:
    artifact = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if artifact is None:
        raise SystemExit("Usage: report.py <artifact-dir>")
    before = counters(artifact / "metrics-before.prom")
    after = counters(artifact / "metrics-after.prom")
    print("metric\tvalue")
    for metric in COUNTERS:
        print(f"{metric}\t{after.get(metric, 0.0) - before.get(metric, 0.0):.0f}")


if __name__ == "__main__":
    main()
