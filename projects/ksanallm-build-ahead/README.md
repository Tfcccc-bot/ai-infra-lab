# KsanaLLM Executor Build-Ahead 快速路径

## 概述

本项目包含我在腾讯 KsanaLLM 实习期间设计的 **Executor Build-Ahead** 机制的核心实现代码。

Build-Ahead 是一种跨 step 的流水线深度优化：允许 Executor 在当前步（step N）的采样结果尚未通过 IPC 回报 Engine 之前，就提前 launch 下一步（step N+1），从而隐藏 IPC 延迟和调度间隙。

## 核心机制

```
慢路径 (max_depth=1):
  Engine Schedule N → IPC → Exec Forward N → IPC → Engine Update N → ...

快路径 (max_depth=2):
  Engine Schedule N → IPC → Exec [Forward N → Sampling N → WriteRing N]
                                ↓ (ring 传递)
  Engine Schedule N+1 (placeholder) → IPC → Exec [Replace N+1 → Forward N+1]
```

### 三层架构

1. **Engine 侧**：max_depth=2 流水线 + placeholder（-1）占位 + 序列化快照防竞态
2. **Executor 侧**：双 slot 乒乓（独立 ModelInput/H2D stream/Sampler buffer）+ Ring Buffer 协议跨步传递采样结果
3. **FastPathController**：启动期硬条件 Gate，13 项一次性检查，排除不兼容功能

### Ring Buffer 协议

每个 slot 维护独立的 device ring buffer，布局为 `int64[1 + 2×max_batch_size]`：
- `ring[0]` = count（本步 batch size）
- 后续依次存 `(req_id, token)` 配对

核心 CUDA kernel：
- **ReplaceLastTokenKernel**：Forward 前读取上一步 ring，回填 placeholder token
- **WriteRingKernel**：Sampling 后将本步结果写入当前 slot ring

## 文件索引

| 目录 | 文件 | 作用 |
|------|------|------|
| `runtime/` | `fast_path_controller.{cpp,h}` | 启动期 Gate，13 项硬条件检查 |
| `runtime/` | `infer_request.{cpp,h}` | 双深度 inflight 队列 + LaunchPlanningTask |
| `runtime/` | `infer_stage.h` | `kFastPathPlaceholderTokenId = -1` 常量 |
| `runtime/` | `schedule_output.{cpp,h}` | IsLaunchable / LaunchScheduleOutput / 序列化快照 |
| `runtime/` | `model_input.h` / `model_output.h` | NewModelInput::slot_index / prev_ring_dev |
| `runtime/` | `executor_infer_request.h` | Executor 侧 inflight 请求定义 |
| `executor/` | `model_runner.{cpp,h}` | 线程模型 / slot 翻转 / Execute 主循环 |
| `executor/` | `model_builder.{cpp,h}` | forwarding_tokens 深拷贝 |
| `executor/` | `inflight_resource.{cpp,h}` | Inflight 资源管理器 |
| `executor/` | `model_sampler.{cpp,h}` | Sampling 调度 |
| `samplers/` | `sampler.{cpp,h}` | 双 ring buffer + WriteRing / Broadcast |
| `layers/` | `replace_last_token.{cu,h}` | ReplaceLastTokenKernel + WriteRingKernel |
| `models/` | `forwarding_context.{cpp,h}` | 双 ModelInput slot 管理 |
| `models/` | `model_input.{cpp,h}` | cur_batch_pairs_tensor 填充 |
| `models/` | `common_model.{cpp,h}` | Forward 中 launch ReplaceLastTokenKernel |
| `configure/` | `schedule_config_parser.{cpp,h}` | enable_executor_build_ahead 配置解析 |
| `batch_scheduler/` | `batch_scheduler.{cpp,h}` | TryToLaunchPlannedScheduleOutput |
| (根) | `build-ahead-mtp-serial.md` | 设计文档：Build-Ahead + MTP 串行方案 |

## 来源

代码提取自 [NumerousLLM](https://git.woa.com/zakwang/NumerousLLM)（腾讯内部 KsanaLLM 代码库），commit `39c8b3c8`。

相关 commit 范围：`8d6dd060` → `39c8b3c8`，涵盖完整的 Build-Ahead 功能演进。

## 许可

本代码仅供个人学习和技术展示使用。版权归腾讯所有。
