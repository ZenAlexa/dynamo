#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOPOLOGY="${1:-global-1x24}"
TRACE="${2:-$ROOT/traces/generated/swe-production-6144x6.jsonl}"
if [[ ! -f "$TRACE" ]]; then
    echo "ERROR: trace not found: $TRACE (run traces/generate_production_trace.sh)" >&2
    exit 2
fi

preserve_run_evidence() {
    local artifact="$1"
    mkdir -p "$artifact/logs" "$artifact/config"
    cp -a "$ROOT/logs/." "$artifact/logs/"
    cp "$ROOT/config/mocker.env" "$ROOT/config/topologies/$TOPOLOGY.env" \
        "$ROOT/generated/infra.env" "$ROOT/generated/prometheus.yml" "$artifact/config/"
    printf '%s\n' "$(realpath "$TRACE")" > "$artifact/config/trace.path"
    sha256sum "$TRACE" > "$artifact/config/trace.sha256"
}

for arm in affinity-off affinity-on; do
    artifact="$ROOT/artifacts/$TOPOLOGY/$arm"
    rm -rf "$artifact"
    "$ROOT/scripts/start.sh" "$TOPOLOGY" "$arm"
    trap '"$ROOT/scripts/stop.sh" "$TOPOLOGY"' EXIT
    "$ROOT/scripts/run_aiperf.sh" "$artifact" "$TRACE"
    preserve_run_evidence "$artifact"

    "$ROOT/scripts/stop.sh" "$TOPOLOGY"
    trap - EXIT
done
