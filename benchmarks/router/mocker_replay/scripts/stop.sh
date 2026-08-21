#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOPOLOGY="${1:-}"
if [[ -z "$TOPOLOGY" && -f "$ROOT/logs/topology" ]]; then
    TOPOLOGY="$(cat "$ROOT/logs/topology")"
fi
if [[ ! -f "$ROOT/config/topologies/${TOPOLOGY}.env" ]]; then
    echo "ERROR: supply the topology used by start.sh" >&2
    exit 2
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    COMPOSE=()
fi

stop_pid() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
}

if [[ -f "$ROOT/logs/frontend.pids" ]]; then
    while read -r pid; do stop_pid "$pid"; done < "$ROOT/logs/frontend.pids"
fi
if [[ -f "$ROOT/logs/mocker.pids" ]]; then
    while read -r pid; do stop_pid "$pid"; done < "$ROOT/logs/mocker.pids"
fi
rm -f "$ROOT/logs/frontend.pids" "$ROOT/logs/frontend.ports" \
    "$ROOT/logs/mocker.pids" "$ROOT/logs/topology"
if (( ${#COMPOSE[@]} > 0 )); then
    compose_env=()
    if [[ -f "$ROOT/generated/infra.env" ]]; then
        compose_env=(--env-file "$ROOT/generated/infra.env")
    fi
    "${COMPOSE[@]}" -f "$ROOT/docker-compose.yml" --project-directory "$ROOT" \
        "${compose_env[@]}" -p "mocker-replay-$TOPOLOGY" down --remove-orphans >/dev/null 2>&1 || true
fi
