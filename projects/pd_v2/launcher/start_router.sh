#!/usr/bin/env bash
# start_router.sh —— 拉起 Router（PD 代理），对外提供 OpenAI 兼容入口
#
# 用法：bash launcher/start_router.sh
# 客户端请求打到 http://<ROUTER_IP>:${ROUTER_PORT}/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/pd-1p1d.env"

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python，请先激活 sglang 环境" >&2; exit 1
fi

echo "[router] 启动 PD 代理：prefill=${PREFILL_IP}:${PREFILL_PORT} decode=${DECODE_IP}:${DECODE_PORT} -> :${ROUTER_PORT}"

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" \
  --decode "http://${DECODE_IP}:${DECODE_PORT}" \
  --host "${ROUTER_IP}" \
  --port "${ROUTER_PORT}"
