<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mocker replay benchmark

A small, OSS-only replay rig for comparing Dynamo's public default KV routing with
session affinity disabled and enabled. It starts NATS, etcd, Prometheus, the
unmodified `dynamo.mocker`, and the unmodified `dynamo.frontend`.

## Prerequisites

- Docker Compose and `curl`.
- Python with this checkout's `dynamo` package (`PYTHON` may select the interpreter).
- AIPerf 0.12.0 on `PATH`.
- Network access, or a populated Hugging Face cache, for the tokenizer/configuration in `config/mocker.env`. Mockers do not load model weights.
- AIConfigurator data for the SGLang calibration in `config/mocker.env`.

Run commands from this directory:

```bash
cd benchmarks/router/mocker_replay
./traces/generate_production_trace.sh
./scripts/start.sh global-1x24 affinity-off
./scripts/run_aiperf.sh artifacts/manual traces/generated/swe-production-6144x6.jsonl
./scripts/stop.sh
./scripts/run_affinity_ladder.sh global-1x24
```

`start.sh` accepts `global-1x24`, `global-2x24`, or `sharded-6x4`, followed by
`affinity-on` or `affinity-off`. `run_affinity_ladder.sh` starts and stops a new
topology for every arm and writes `artifacts/<topology>/<arm>/`.

Router-side flow control is enabled by default through
`config/router-queue.yaml`: FCFS queueing begins only when every eligible worker
exceeds 90% of its advertised maximum batched-token capacity, with a limit of
256 queued requests per discovered worker. Override the complete public policy
configuration with a relative or absolute path when comparing another policy:

```bash
ROUTER_POLICY_CONFIG=path/to/policy.yaml ./scripts/start.sh global-1x24 affinity-on
```

Prometheus is available on the topology-specific `PROMETHEUS_PORT` printed by
`start.sh`. Each metrics snapshot includes every frontend and mocker endpoint.
The ladder also preserves the resolved policy, process logs, generated scrape
configuration, topology settings, and trace checksum under each arm's artifact
directory before teardown. `scripts/report.py artifacts/manual` prints a compact
table from the public `/metrics` snapshots.
