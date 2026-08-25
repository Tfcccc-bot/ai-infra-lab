#!/usr/bin/env bash
# env_probe.sh — PD v2 M0 环境探测脚本
#
# 目标：把 GPU / PCIe / NUMA / NIC / HCA / RDMA / Mooncake 的拓扑一次性打出来，
#       并生成 topology/topology-<hostname>.json 供后续拓扑感知路由使用。
#
# 用法：bash launcher/env_probe.sh
# 输出：stdout 摘要 + topology/topology-<hostname>.json

set -uo pipefail

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/topology"
OUT_JSON="${OUT_DIR}/topology-${HOSTNAME_SHORT}.json"

mkdir -p "${OUT_DIR}"

# ---- helper: 探测一个命令是否存在 ----
have() { command -v "$1" >/dev/null 2>&1; }

# ---- helper: 把多行文本转成 JSON 字符串数组 ----
lines_to_json_array() {
  # 读取 stdin 的非空行，输出 JSON 数组字符串
  python3 -c 'import sys,json; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.strip()]))' 2>/dev/null \
    || echo '[]'
}

echo "== PD v2 环境探测 =="
echo "主机: ${HOSTNAME_SHORT}"
echo "时间: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo

# ---------------- 软件版本 ----------------
echo "---- 软件版本 ----"
PY_VER="$(python3 --version 2>&1 || echo n/a)"; echo "python: ${PY_VER}"
CUDA_VER="$(nvcc --version 2>/dev/null | grep -i 'release' | head -1 || echo n/a)"; echo "cuda:   ${CUDA_VER}"
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || echo n/a)"; echo "torch:  ${TORCH_VER}"
SGL_VER="$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || echo n/a)"; echo "sglang: ${SGL_VER}"
if have pip; then
  MOONCAKE_VER="$(pip show mooncake-transfer-engine 2>/dev/null | grep -i '^Version' | awk '{print $2}' || echo n/a)"
  echo "mooncake: ${MOONCAKE_VER}"
fi
RDMA_CORE_VER="$(dpkg -s librdmacm1 2>/dev/null | grep -i '^Version' | awk '{print $2}' || rpm -q rdma-core 2>/dev/null || echo n/a)"; echo "rdma-core: ${RDMA_CORE_VER}"

# ---------------- GPU ----------------
echo
echo "---- GPU ----"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,pci.domain --format=csv,noheader 2>/dev/null || true
  GPU_JSON="$(nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,pci.domain --format=csv,noheader 2>/dev/null | lines_to_json_array)"
  TOPO="$(nvidia-smi topo -m 2>/dev/null || true)"
  echo "--- GPU 拓扑 (nvidia-smi topo -m) ---"
  echo "${TOPO}"
  TOPO_JSON="$(echo "${TOPO}" | lines_to_json_array)"
elif have rocm-smi; then
  rocm-smi --showproductname --showmeminfo vram 2>/dev/null || true
  GPU_JSON='[]'; TOPO_JSON='[]'
else
  echo "未检测到 GPU 工具 (nvidia-smi / rocm-smi)"
  GPU_JSON='[]'; TOPO_JSON='[]'
fi

# ---------------- PCIe ----------------
echo
echo "---- PCIe (GPU/NIC/HCA 相关) ----"
if have lspci; then
  lspci 2>/dev/null | grep -Ei 'nvidia|amd|mellanox|infiniband|ethernet controller|rdma|hca' || echo "  (无匹配的 GPU/NIC/HCA PCI 设备)"
  PCIE_JSON="$(lspci 2>/dev/null | grep -Ei 'nvidia|amd|mellanox|infiniband|ethernet controller|rdma|hca' | lines_to_json_array)"
else
  PCIE_JSON='[]'
fi

# ---------------- NUMA ----------------
echo
echo "---- NUMA ----"
if have numactl; then
  numactl --hardware 2>/dev/null || true
  NUMA_JSON="$(numactl --hardware 2>/dev/null | lines_to_json_array)"
else
  echo "  numactl 未安装；尝试 lscpu"
  lscpu 2>/dev/null | grep -iE 'numa|node\(s\)|socket' || echo "  (无 NUMA 信息)"
  NUMA_JSON="$(lscpu 2>/dev/null | grep -iE 'numa|node\(s\)|socket' | lines_to_json_array)"
fi

# ---------------- NIC / HCA / RDMA ----------------
echo
echo "---- NIC ----"
if have ip; then
  ip -brief link show 2>/dev/null || ifconfig -a 2>/dev/null || true
  NIC_JSON="$(ip -brief link show 2>/dev/null | lines_to_json_array)"
else
  NIC_JSON='[]'
fi

echo
echo "---- HCA / RDMA ----"
if have ibv_devinfo; then
  ibv_devinfo 2>/dev/null || true
  HCA_JSON="$(ibv_devinfo 2>/dev/null | grep -iE 'hca_id|port|state|link_layer' | lines_to_json_array)"
else
  echo "  ibv_devinfo 未安装"
  HCA_JSON='[]'
fi
if have rdma; then
  rdma link show 2>/dev/null || true
  RDMA_JSON="$(rdma link show 2>/dev/null | lines_to_json_array)"
else
  RDMA_JSON='[]'
fi

# ---------------- 写 JSON ----------------
python3 - "${OUT_JSON}" "${HOSTNAME_SHORT}" "${GPU_JSON}" "${TOPO_JSON}" "${PCIE_JSON}" "${NUMA_JSON}" "${NIC_JSON}" "${HCA_JSON}" "${RDMA_JSON}" <<'PYEOF' 2>/dev/null
import sys, json
out, host = sys.argv[1], sys.argv[2]
gpu, topo, pcie, numa, nic, hca, rdma = sys.argv[3:10]
doc = {
    "hostname": host,
    "probed_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "gpu": json.loads(gpu),
    "gpu_topology": json.loads(topo),
    "pcie": json.loads(pcie),
    "numa": json.loads(numa),
    "nic": json.loads(nic),
    "hca": json.loads(hca),
    "rdma": json.loads(rdma),
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
print("已写入拓扑 JSON:", out)
PYEOF

echo
echo "完成。若需更多字段（IB 设备与 NUMA 映射、带宽实测），见 topology/README.md"
