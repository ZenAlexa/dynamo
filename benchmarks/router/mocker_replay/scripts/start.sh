#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TOPOLOGY="${1:-}"
AFFINITY="${2:-affinity-off}"
TOPOLOGY_FILE="$ROOT/config/topologies/${TOPOLOGY}.env"

if [[ ! -f "$TOPOLOGY_FILE" ]]; then
    echo "ERROR: topology must be global-1x24, global-2x24, or sharded-6x4" >&2
    exit 2
fi
if [[ "$AFFINITY" != "affinity-on" && "$AFFINITY" != "affinity-off" ]]; then
    echo "ERROR: affinity must be affinity-on or affinity-off" >&2
    exit 2
fi
command -v "$PYTHON" >/dev/null || { echo "ERROR: Python not found: $PYTHON" >&2; exit 2; }

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    echo "ERROR: Docker Compose is required" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$ROOT/config/mocker.env"
# shellcheck disable=SC1090
source "$TOPOLOGY_FILE"
MOCKER_SGLANG_GENERATE="${MOCKER_SGLANG_GENERATE:-0}"
mkdir -p "$ROOT/generated" "$ROOT/logs" "$ROOT/artifacts"

if [[ -f "$ROOT/logs/topology" ]]; then
    "$ROOT/scripts/stop.sh" "$(cat "$ROOT/logs/topology")" >/dev/null 2>&1 || true
fi
"$ROOT/scripts/stop.sh" "$TOPOLOGY" >/dev/null 2>&1 || true

cat > "$ROOT/generated/infra.env" <<EOF
NATS_PORT=$NATS_PORT
NATS_MONITORING_PORT=$NATS_MONITORING_PORT
ETCD_PORT=$ETCD_PORT
PROMETHEUS_PORT=$PROMETHEUS_PORT
EOF

{
    printf 'global:\n  scrape_interval: 5s\nscrape_configs:\n'
    printf '  - job_name: dynamo-frontend\n    static_configs:\n      - targets:\n'
    for ((frontend = 0; frontend < FRONTEND_COUNT; frontend++)); do
        printf '          - "host.docker.internal:%s"\n' "$((FRONTEND_PORT_BASE + frontend))"
    done
    printf '  - job_name: dynamo-mocker\n    static_configs:\n      - targets:\n'
    for ((worker = 0; worker < WORKER_COUNT; worker++)); do
        printf '          - "host.docker.internal:%s"\n' "$((SYSTEM_PORT_BASE + worker))"
    done
} > "$ROOT/generated/prometheus.yml"

echo "$TOPOLOGY" > "$ROOT/logs/topology"
cleanup_failed_start() {
    local status=$?
    if (( status != 0 )); then
        "$ROOT/scripts/stop.sh" "$TOPOLOGY" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_failed_start EXIT

export NATS_PORT NATS_MONITORING_PORT ETCD_PORT PROMETHEUS_PORT
"${COMPOSE[@]}" -f "$ROOT/docker-compose.yml" --project-directory "$ROOT" \
    --env-file "$ROOT/generated/infra.env" -p "mocker-replay-$TOPOLOGY" up -d --build

wait_http() {
    local url="$1" label="$2" pid="${3:-}"
    for _ in $(seq 1 120); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $label exited; inspect $ROOT/logs" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 1 "$url" >/dev/null; then
            return
        fi
        sleep 1
    done
    echo "ERROR: $label was not ready at $url; inspect $ROOT/logs" >&2
    exit 1
}

all_mockers_alive() {
    while read -r mocker_pid; do
        kill -0 "$mocker_pid" 2>/dev/null || return 1
    done < "$ROOT/logs/mocker.pids"
}

wait_model() {
    local url="$1" label="$2" pid="$3" response
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null || ! all_mockers_alive; then
            echo "ERROR: $label or a mocker exited; inspect $ROOT/logs" >&2
            exit 1
        fi
        response="$(curl --fail --silent --max-time 1 "$url" 2>/dev/null || true)"
        if [[ "$response" == *"$MODEL_NAME"* ]]; then
            return
        fi
        sleep 1
    done
    echo "ERROR: $label did not advertise $MODEL_NAME; inspect $ROOT/logs" >&2
    exit 1
}

wait_http "http://127.0.0.1:$NATS_MONITORING_PORT/varz" "NATS"
wait_http "http://127.0.0.1:$ETCD_PORT/health" "etcd"

BASE_NAMESPACE="mocker-replay-$TOPOLOGY"
WORKERS_PER_SHARD=$((WORKER_COUNT / SHARD_COUNT))
: > "$ROOT/logs/mocker.pids"
: > "$ROOT/logs/mocker.ports"

mocker_generate_args=()
frontend_generate_env=()
if [[ "$MOCKER_SGLANG_GENERATE" == 1 ]]; then
    mocker_generate_args=(--sglang-generate)
    frontend_generate_env=(DYN_SGLANG_ENABLE_GENERATE=1)
fi

for ((worker = 0; worker < WORKER_COUNT; worker++)); do
    shard=$((worker / WORKERS_PER_SHARD))
    worker_namespace="$BASE_NAMESPACE"
    if (( SHARD_COUNT > 1 )); then
        worker_namespace="$BASE_NAMESPACE-shard-$shard"
    fi
    env \
        DYN_NAMESPACE="$worker_namespace" \
        DYN_DISCOVERY_BACKEND=etcd \
        ETCD_ENDPOINTS="http://127.0.0.1:$ETCD_PORT" \
        NATS_SERVER="nats://127.0.0.1:$NATS_PORT" \
        DYN_REQUEST_PLANE=tcp \
        DYN_EVENT_PLANE=nats \
        DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + worker))" \
        DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]' \
        "$PYTHON" -m dynamo.mocker \
            --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" --num-workers 1 \
            --engine-type "$MOCKER_ENGINE_TYPE" --block-size "$MOCKER_BLOCK_SIZE" \
            --max-num-seqs "$MOCKER_MAX_NUM_SEQS" \
            --max-num-batched-tokens "$MOCKER_MAX_NUM_BATCHED_TOKENS" \
            --num-gpu-blocks-override "$MOCKER_NUM_GPU_BLOCKS" \
            --sglang-page-size "$MOCKER_BLOCK_SIZE" \
            --sglang-max-prefill-tokens "$SGLANG_MAX_PREFILL_TOKENS" \
            --sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE" \
            --sglang-schedule-policy fcfs --sglang-schedule-conservativeness 1.0 \
            --aic-perf-model --aic-system "$AIC_SYSTEM" --aic-backend "$AIC_BACKEND" \
            --aic-backend-version "$AIC_BACKEND_VERSION" --aic-tp-size "$AIC_TP_SIZE" \
            --aic-moe-tp-size "$AIC_MOE_TP_SIZE" --aic-moe-ep-size "$AIC_MOE_EP_SIZE" \
            --aic-attention-dp-size "$AIC_ATTENTION_DP_SIZE" \
            "${mocker_generate_args[@]}" \
        > "$ROOT/logs/mocker-$worker.log" 2>&1 &
    echo "$!" >> "$ROOT/logs/mocker.pids"
    echo "$((SYSTEM_PORT_BASE + worker))" >> "$ROOT/logs/mocker.ports"

done

# Enable bounded router-side flow control by default. Callers can override this
# with a complete public policy-class configuration.
ROUTER_POLICY_CONFIG="${ROUTER_POLICY_CONFIG:-$ROOT/config/router-queue.yaml}"

policy_args=()
if [[ -n "${ROUTER_POLICY_CONFIG:-}" ]]; then
    if [[ ! -f "$ROUTER_POLICY_CONFIG" ]]; then
        echo "ERROR: ROUTER_POLICY_CONFIG does not exist: $ROUTER_POLICY_CONFIG" >&2
        exit 2
    fi
    policy_args=(--router-policy-config "$(realpath "$ROUTER_POLICY_CONFIG")")
    cp "$(realpath "$ROUTER_POLICY_CONFIG")" "$ROOT/logs/router-policy.yaml"
fi

: > "$ROOT/logs/frontend.pids"
: > "$ROOT/logs/frontend.ports"
for ((frontend = 0; frontend < FRONTEND_COUNT; frontend++)); do
    frontend_namespace="$BASE_NAMESPACE"
    if (( SHARD_COUNT > 1 )); then
        frontend_namespace="$BASE_NAMESPACE-shard-$frontend"
    fi
    frontend_args=(
        --namespace "$frontend_namespace"
        --model-name "$MODEL_NAME"
        --router-mode kv
        --http-port "$((FRONTEND_PORT_BASE + frontend))"
        --kv-cache-block-size "$MOCKER_BLOCK_SIZE"
        --router-min-initial-workers "$WORKERS_PER_SHARD"
    )
    if [[ "$AFFINITY" == "affinity-on" ]]; then
        frontend_args+=(--router-session-affinity-ttl-secs 600)
    fi
    if [[ "$ROUTER_REPLICA_SYNC" == 1 ]]; then
        frontend_args+=(--router-replica-sync)
    fi
    env \
        DYN_DISCOVERY_BACKEND=etcd \
        ETCD_ENDPOINTS="http://127.0.0.1:$ETCD_PORT" \
        NATS_SERVER="nats://127.0.0.1:$NATS_PORT" \
        DYN_REQUEST_PLANE=tcp \
        DYN_EVENT_PLANE=nats \
        DYN_SYSTEM_PORT="$((SYSTEM_PORT_BASE + WORKER_COUNT + frontend))" \
        "${frontend_generate_env[@]}" \
        "$PYTHON" -m dynamo.frontend "${frontend_args[@]}" "${policy_args[@]}" \
        > "$ROOT/logs/frontend-$frontend.log" 2>&1 &
    echo "$!" >> "$ROOT/logs/frontend.pids"
    echo "$((FRONTEND_PORT_BASE + frontend))" >> "$ROOT/logs/frontend.ports"
done

frontend=0
while read -r pid; do
    wait_model "http://127.0.0.1:$((FRONTEND_PORT_BASE + frontend))/v1/models" \
        "frontend $frontend" "$pid"
    frontend=$((frontend + 1))
done < "$ROOT/logs/frontend.pids"

trap - EXIT
echo "Ready: topology=$TOPOLOGY affinity=$AFFINITY frontends=$FRONTEND_COUNT workers=$WORKER_COUNT prometheus=http://127.0.0.1:$PROMETHEUS_PORT"
