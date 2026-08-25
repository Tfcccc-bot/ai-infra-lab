#!/usr/bin/env bash
# start_prefill.sh —— 拉起 Prefill 实例（跨机，多节点时在每个节点各跑一次）
#
# 用法：
#   bash launcher/start_prefill.sh 0     # 节点 0（master）
#   bash launcher/start_prefill.sh 1     # 节点 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/pd-1p1d.env"

NODE_RANK="${1:-0}"

# 前置依赖断言（失败给可读原因，而不是堆栈）
if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python，请先激活 sglang 环境" >&2; exit 1
fi
if [[ "${TRANSFER_BACKEND}" == "mooncake" ]]; then
  if ! command -v ibv_devinfo >/dev/null 2>&1 && [[ "${IB_DEVICE}" != "CHANGEME" ]]; then
    echo "[WARN] 未检测到 ibv_devinfo，RDMA 可能不可用；确认 IB_DEVICE 是否正确" >&2
  fi
fi

# 拼装可选参数
IB_DEVICE_ARG=()
if [[ "${TRANSFER_BACKEND}" == "mooncake" && -n "${IB_DEVICE:-}" && "${IB_DEVICE}" != "CHANGEME" ]]; then
  IB_DEVICE_ARG=("--disaggregation-ib-device" "${IB_DEVICE}")
fi

MOE_ARG=()
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
  MOE_ARG=("--moe-a2a-backend" "${MOE_A2A_BACKEND}")
fi

echo "[prefill] 节点 rank=${NODE_RANK} 启动，模型=${MODEL_PATH} 后端=${TRANSFER_BACKEND}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  ${TRUST_REMOTE_CODE} \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "${TRANSFER_BACKEND}" \
  "${IB_DEVICE_ARG[@]}" \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --host "${PREFILL_IP}" \
  --port "${PREFILL_PORT}" \
  --dist-init-addr "${PREFILL_MASTER_IP}:${DIST_INIT_PORT}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --tp-size "${TP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  ${ENABLE_DP_ATTENTION} \
  "${MOE_ARG[@]}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
