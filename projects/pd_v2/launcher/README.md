# launcher/

启停脚本 + 环境检查。

- `env_probe.sh` —— M0 环境探测，生成 `topology/topology-<hostname>.json`（**先跑这个**）
- 后续补充：`start_pd.sh`（拉起 Router + Prefill + Decode）、`stop_pd.sh`、`check_env.sh`（前置依赖断言）

## 约定

- 启动脚本要能 `bash launcher/start_pd.sh` 一键复现 M1 基线，参数从 `configs/` 读取，不散落在脚本里。
- 涉及 Mooncake/RDMA 的启动前，先断言 `ibv_devinfo` / `rdma link` 状态，失败时给出可读原因而不是堆栈。
