#!/usr/bin/env bash
# check_env.sh —— 跨机 PD 启动前的前置依赖断言
#
# 用法：bash launcher/check_env.sh
# 退出码 0 表示可启动；非 0 表示有缺口，按输出逐一补齐。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/pd-1p1d.env"

FAIL=0
ok()   { echo "  [OK]   $1"; }
warn() { echo "  [WARN] $1"; }
bad()  { echo "  [FAIL] $1"; FAIL=1; }

echo "== 检查配置 =="
if [[ "${MODEL_PATH}" == CHANGEME* ]]; then
  bad "MODEL_PATH 仍是占位符，请填 deepseekv4pro 的真实路径/HF id"
elif [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* ]]; then
  if [[ -e "${MODEL_PATH}" ]]; then ok "模型本地路径存在: ${MODEL_PATH}"
  else bad "模型本地路径不存在: ${MODEL_PATH}"; fi
else
  ok "模型走 HF 下载: ${MODEL_PATH}（需网络或 HF_HOME 缓存）"
fi
[[ "${PREFILL_IP}" == "10.0.0.1" ]] && warn "PREFILL_IP 仍是示例值，请改成真实 IP" || ok "PREFILL_IP=${PREFILL_IP}"
[[ "${DECODE_IP}" == "10.0.0.2" ]] && warn "DECODE_IP 仍是示例值，请改成真实 IP" || ok "DECODE_IP=${DECODE_IP}"

echo "== 检查运行时 =="
command -v python >/dev/null 2>&1 && ok "python: $(python --version 2>&1)" || bad "未找到 python"
python -c 'import sglang; print("  [OK]   sglang", sglang.__version__)' 2>/dev/null || bad "sglang 未安装/不可导入"

echo "== 检查传输后端 =="
case "${TRANSFER_BACKEND}" in
  mooncake|mooncake_tcp)
    python -c 'import mooncake' 2>/dev/null && ok "mooncake 已安装" || bad "mooncake 未安装（uv pip install mooncake-transfer-engine）"
    if [[ "${TRANSFER_BACKEND}" == "mooncake" ]]; then
      command -v ibv_devinfo >/dev/null 2>&1 && ok "ibv_devinfo 存在" || warn "ibv_devinfo 不存在，RDMA 可能不可用（无 RDMA 请改用 nixl 或 mooncake_tcp）"
      command -v rdma >/dev/null 2>&1 && { rdma link show 2>/dev/null | head -5; } || true
      if [[ "${IB_DEVICE:-}" != "CHANGEME" && -n "${IB_DEVICE:-}" ]]; then
        ibv_devinfo 2>/dev/null | grep -q "${IB_DEVICE}" && ok "IB 设备 ${IB_DEVICE} 存在" || warn "未在 ibv_devinfo 中匹配到 ${IB_DEVICE}"
      fi
    fi
    ;;
  nixl)
    python -c 'import nixl' 2>/dev/null && ok "nixl 已安装" || bad "nixl 未安装（pip install nixl）"
    ;;
  *)
    warn "未知传输后端: ${TRANSFER_BACKEND}"
    ;;
esac

echo "== 检查 MoE 后端 =="
if [[ -n "${MOE_A2A_BACKEND:-}" ]]; then
  python -c 'import deep_ep' 2>/dev/null && ok "deep_ep 已安装（moe-a2a-backend=deepep）" || warn "deep_ep 未安装；非 MoE 模型请置空 MOE_A2A_BACKEND"
fi

echo
if [[ "${FAIL}" == 0 ]]; then
  echo "结论：可启动（WARN 项按需处理）。"
else
  echo "结论：存在 FAIL 项，先补齐再启动。"
fi
exit "${FAIL}"
