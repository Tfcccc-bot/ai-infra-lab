# configs/

P/D/Router 与拓扑配置。SGLang 原生 PD 参数（`--disaggregation-mode`、`--disaggregation-transfer-backend`、Mooncake/NIXL 相关）在这里固化，避免每次手敲。

## 文件

- `pd-1p1d.env` —— **跨机 1P1D**（Prefill/Decode 分离）+ DeepSeek 系列（MoE+MLA）的中心配置。
  所有 `launcher/start_*.sh` 通过 `source` 读取它，改这一份即可。
  - 必填：`MODEL_PATH`（deepseekv4pro 真实路径）、`PREFILL_IP` / `DECODE_IP`、`IB_DEVICE`
  - 并行：`NNODES` / `TP_SIZE` / `DP_SIZE`（按模型显存重算，DeepSeek-V3 参考 TP16+DP8）
- `topology/topology-*.json` —— 由 `launcher/env_probe.sh` 生成（不入库）

## 约定

- 不要写死 GPU 数量 / 端口 / IB 设备为"全局唯一真理"——M3 的目标正是用拓扑自动生成，而非静态列表。
- 新增模型/拓扑场景时，复制 `pd-1p1d.env` 改名，别在原文件里堆多个互斥配置。
