# tests/

基于 SGLang `fake` backend 的多 scheduler 状态机与故障测试（M2）。

## 原则

- **自行实现**测试场景，不搬 KsanaLLM 内部测试代码。
- 覆盖 room 生命周期：Created → Bootstrapping → DestPublished → Transferring → Committed / Failed / Aborted。
- 覆盖：P 先到、D 先到、bootstrap 超时、传输超时、某 TP rank 失败、重复 room、abort 与迟到 ACK、decode retraction。

## 不变量（每条用例都要断言）

- 所有 rank 状态一致
- 资源只释放一次
- 失败不污染后续复用 room
- 无孤儿 prealloc slot

## 计划落地

- `test_room_lifecycle.py`、`test_failure_injection.py`、`test_concurrency.py`
- 随机故障注入 + 1000 次随机调度压测
