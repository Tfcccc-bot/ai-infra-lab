# KsanaLLM Executor Build-Ahead 🚀

> 跨 step 流水线深度优化：隐藏 IPC 延迟与调度间隙
>
> 腾讯 KsanaLLM 实习核心产出

---

## 概述

**Executor Build-Ahead** 是腾讯 KsanaLLM 实习期间设计的跨 step 流水线深度优化机制：允许 Executor 在当前步（step N）的采样结果尚未通过 IPC 回报 Engine 之前，就提前 launch 下一步（step N+1），从而隐藏 IPC 延迟和调度间隙。

核心机制：

- **双 Slot 乒乓**：两个 slot 各自独立 ModelInput / H2D stream / Sampler buffer，Preprocess 与 Forward 并行
- **Ring Buffer 协议**：`int64[1 + 2×max_batch_size]` 布局，ReplaceLastTokenKernel + WriteRingKernel 跨步传递采样结果
- **Placeholder 机制**：Engine 用 `kFastPathPlaceholderTokenId = -1` 占位，Executor 侧 Forward 前回填
- **FastPathController**：启动期 13 项硬条件 Gate，不做运行时判断

## 项目结构

```
ai-infra-lab/
└── projects/
    └── ksanallm-build-ahead/     # KsanaLLM Executor Build-Ahead 核心实现
```

## 快速链接

- [核心实现与文件索引](projects/ksanallm-build-ahead/README.md)
- [设计文档：Build-Ahead + MTP 串行方案](projects/ksanallm-build-ahead/build-ahead-mtp-serial.md)

## 许可

代码仅供个人学习和技术展示使用，版权归腾讯所有。
