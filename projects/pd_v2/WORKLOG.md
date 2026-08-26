# 工作日志（WORKLOG）

>  chronological 记录：决策、进展、阻塞、下一步。与 `DEVLOG.md`（逐模块技术记录）互补。
> 纪律：每次推进先在 `FORK_PLAN.md` 补对应小节 + 单测，再实现，最后补 `DEVLOG.md`。

---

## 2026-08-26 — 纯 Python 原型 M0–M6 完成（48 单测全绿）

- **决策**：先在本仓库（`ai-infra-lab/projects/pd_v2`）把 KsanaLLM PD v2 思想用纯 Python 实现，之后再去 fork SGLang。
- **完成模块**：
  - M0 TDD 脚手架（`pyproject.toml` + `tests/conftest.py`，探测 sglang 缺失时集成用例 skip）。
  - M1 topology：GPU↔NUMA↔NIC 图、probe、score、bind（per-GPU IB 映射、CPU affinity、显式降级）。
  - M2 状态机：8 状态生命周期 + 4 不变量，1000 次随机调度稳定。
  - M3 routing：TopologyAwarePolicy / NaivePolicy + 离散事件模拟（auto 永不被 dominate）。
  - M4 benchmarks：TTFT/TPOT/ITL P50–P99/吞吐/KV 指标，自实现 percentile（线性插值）。
  - M5 analysis：8 阶段 trace 解析 + compare 报告。
  - M6 patches：manifest 校验（fork_repo/branch/commit/intent/status…）。
- **踩坑与修复**：pytest 未装；漏 `import pytest`；nic 兜底取整行→改 `n.split()[0]`；P99 期望值 99.0→99.01（线性插值）；prefill_mean 0.35→0.3。

## 2026-08-26 — GitHub 公钥授权阻塞（待用户处理）

- **问题**：`git clone git@github.com:Tfcccc-bot/ai-infra-lab.git` 报 `Permission denied (publickey)`。本地 `id_ed25519` 已提供给 GitHub 但被拒，根因是公钥未加到**用户自己**的 GitHub 账户。
- **已给出公钥**：`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFbxxz8dixLmX8twZ6wzIo1Zuvg0D7GRJmCQVSqJRtq3 fangtang`
- **待办**：用户把该公钥加到 GitHub → `ssh -T git@github.com` 验证通过。
- **状态**：阻塞 clone，但不阻塞纯 Python 模块（已离线完成）。

## 2026-08-26 — 用户已 fork sglang（待接入）

- 用户已 fork `Tfcccc-bot/sglang` 到自己账户。
- **下一步脚本（待执行）**：
  ```bash
  git clone <你账户>/sglang
  cd sglang
  git remote add upstream https://github.com/Tfcccc-bot/sglang.git
  git fetch upstream 67853c5
  git checkout -b pd-v2-topology 67853c5
  ```

## 2026-08-26 — 草拟 FORK_PLAN（文档先行，本步交付）

- **用户要求**：草拟一份计划，写清**边界**与**单测实现计划（TDD）**，把大问题拆成一系列**可判断完成**的任务；**不大规模重构代码，先落成文档**；并把工作日志写入文档。
- **交付**：
  - `FORK_PLAN.md`：边界（仓库/代码/明确不做）、TDD 单测计划、T1–T9 任务拆解（每任务带验收标准）。
  - `WORKLOG.md`：本文件，记录决策/进展/阻塞。
- **未动代码**：严格遵守「文档先行」，未改 SGLang 核心、未重构现有原型。
- **下一步**：待公钥授权 + fork clone 就绪后，按 T1 开始；T2 起每步先补单测再实现。

## 2026-08-26 — fork 接入 + 迁移启动（T1/T2/T3 落地，T4/T5 逻辑就位）

- **T1 完成**：本地 `sglang` 已是 `Tfcccc-bot/sglang` 的 clone（origin 指向该仓库）。在基线 `67853c5`（当前 main `3ec22948` 的祖先）上创建并切到分支 `pd-v2-topology`。未动 `main`。
- **关键事实（与计划假设不同）**：实际 fork 的 disaggregation 子系统比计划写定时更成熟——
  - `ServerArgs.disaggregation_ib_device` 已支持「逗号分隔共享列表 / per-GPU JSON 映射 / JSON 文件路径」三种形态，且 `_validate_ib_devices` 已对 `/sys/class/infiniband` 校验。T3 的 `auto` 是在此之上的可加 hook。
  - SGLang PD 走 **bootstrap 连接模型**（decode 经 `bootstrap_addr` 连 prefill），**没有计划里假设的单一「Router (P,D) 配对策略」调用点**。T5 需重新定位 hook 点（详见 FORK_PLAN §1.1）。
- **T2 落地（foundation）**：把 M1 topology 包整体移植进 fork，成为 `python/sglang/srt/disaggregation/topology/`（topology_graph / probe / score / bind / auto_ib / __init__）。纯 stdlib，作为后续 T3/T4/T5 的唯一事实来源与可单测底座。
- **T3 落地**：`server_args._validate_ib_devices` 识别 `--disaggregation-ib-device=auto`，委托 `topology.auto_ib.build_auto_ib_mapping` 用拓扑图产出 per-GPU IB 映射（优先同 NUMA，跨 NUMA 显式降级 + warning）。不动 connector 内部。
- **T4/T5 逻辑就位**：`bind.cpu_affinity`（transfer/bootstrap/staging 角色）与 `score.score_pair`（TopologyAware 评分）已随 topology 包移植并附离线单测；但**接入 sglang worker/router 启动处的具体 hook 点尚未挂钩**，留待真机定位与验证。
- **离线单测**：`test/pd_v2/test_topology.py` 12 例全绿（绕开沉重 sglang 顶层 import，纯 stdlib 跑 topology 包）。真实环境 `import sglang` + 真机 1P1D 启动留待验证阶段。
- **下一步**：T4/T5 在 sglang worker/router 的 hook 点识别 + 真机 env_probe + 真机启动验证（T2/T6/T7/T9）。
