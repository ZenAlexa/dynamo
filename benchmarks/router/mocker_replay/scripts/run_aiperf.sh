#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT_DIR="${1:-}"
TRACE="${2:-$ROOT/traces/generated/swe-production-6144x6.jsonl}"
if [[ -z "$ARTIFACT_DIR" ]]; then
    echo "Usage: $0 <artifact-dir> [trace.jsonl]" >&2
    exit 2
fi
if [[ ! -f "$TRACE" ]]; then
    echo "ERROR: trace not found: $TRACE (run traces/generate_production_trace.sh)" >&2
    exit 2
fi
if [[ ! -f "$ROOT/logs/frontend.ports" ]]; then
    echo "ERROR: no running topology; start it with scripts/start.sh" >&2
    exit 2
fi
command -v aiperf >/dev/null || { echo "ERROR: AIPerf 0.12.0 is required on PATH" >&2; exit 2; }
if [[ "$(aiperf --version)" != "0.12.0" ]]; then
    echo "ERROR: AIPerf 0.12.0 is required; found $(aiperf --version)" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$ROOT/config/mocker.env"
mapfile -t FRONTEND_PORTS < "$ROOT/logs/frontend.ports"
mapfile -t MOCKER_PORTS < "$ROOT/logs/mocker.ports"

mkdir -p "$ARTIFACT_DIR"

snapshot_metrics() {
    local destination="$1"
    : > "$destination"
    for port in "${FRONTEND_PORTS[@]}" "${MOCKER_PORTS[@]}"; do
        printf '# source_endpoint http://127.0.0.1:%s/metrics\n' "$port" >> "$destination"
        curl --fail --silent "http://127.0.0.1:$port/metrics" >> "$destination"
        printf '\n' >> "$destination"
    done
}

urls=()
for port in "${FRONTEND_PORTS[@]}"; do
    urls+=(--url "http://127.0.0.1:$port")
done

snapshot_metrics "$ARTIFACT_DIR/metrics-before.prom"
aiperf profile --model "$MODEL_NAME" --tokenizer "$TOKENIZER" \
    "${urls[@]}" --endpoint-type chat --streaming \
    --input-file "$TRACE" --custom-dataset-type mooncake_trace \
    --isl-block-size "$TRACE_BLOCK_SIZE" --fixed-schedule --fixed-schedule-auto-offset \
    --session-header X-Dynamo-Session-ID --concurrency "$AIPERF_CONCURRENCY" \
    --use-server-token-count --random-seed 20260821 \
    --artifact-dir "$ARTIFACT_DIR" --ui simple
snapshot_metrics "$ARTIFACT_DIR/metrics-after.prom"
printf 'aiperf_version=%s\n' "$(aiperf --version)" > "$ARTIFACT_DIR/versions.txt"
python3 "$ROOT/scripts/report.py" "$ARTIFACT_DIR"
