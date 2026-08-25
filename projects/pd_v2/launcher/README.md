# launcher/

启停脚本 + 环境检查。

## 文件

- `env_probe.sh` —— M0 环境探测，生成 `topology/topology-<hostname>.json`（**先跑这个**）
- `check_env.sh` —— 启动前前置依赖断言（模型路径、sglang/mooncake/nixl/deep_ep、RDMA、IB 设备）
- `start_prefill.sh [node_rank]` —— 拉起 Prefill 实例（跨机时每节点跑一次）
- `start_decode.sh [node_rank]` —— 拉起 Decode 实例
- `start_router.sh` —— 拉起 Router（PD 代理，OpenAI 兼容入口）

## 跨机 1P1D 启动顺序（每台机器）

```bash
# 1) 先探测环境
bash launcher/env_probe.sh
# 2) 断言依赖
bash launcher/check_env.sh
# 3) 按角色启动（node_rank 从 0 开始）
bash launcher/start_prefill.sh 0    # prefill 节点 0（master）
bash launcher/start_prefill.sh 1    # prefill 节点 1
bash launcher/start_decode.sh  0    # decode 节点 0（master）
bash launcher/start_decode.sh  1    # decode 节点 1
bash launcher/start_router.sh       # router（可与 prefill/decode 同机或独立）
```

客户端打 `http://<ROUTER_IP>:8000/v1`。

## 约定

- 启动脚本参数从 `configs/` 读取，不散落在脚本里。
- 涉及 Mooncake/RDMA 的启动前，先断言 `ibv_devinfo` / `rdma link` 状态，失败给可读原因而非堆栈。
- DeepSeek（MLA）模型**不要**开 `SGLANG_DISAGG_STAGING_BUFFER`。
