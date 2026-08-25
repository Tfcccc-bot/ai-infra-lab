# configs/

P/D/Router 与拓扑配置。SGLang 原生的 PD 参数（`--disaggregation-mode`、`--disaggregation-transfer-backend`、Mooncake/NIXL 相关）先以这里的启动配置固化，避免每次手敲。

规划中要落地的文件：

- `sglang-pd-qwen3-8b-1p1d.sh` —— 1P1D + Router 的最小可复现启动（M1 基线）
- `topology-*.json` —— 由 `launcher/env_probe.sh` 生成的拓扑（也可放 topology/）
- 后续 M3 的 per-GPU `--disaggregation-ib-device` 映射样例

> 注意：不要把 GPU 数量、端口、IB 设备写死成“全局唯一真理”，M3 的目标正是用拓扑自动生成而非静态列表。
