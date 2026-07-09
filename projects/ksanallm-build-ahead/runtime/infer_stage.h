/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ksana_llm {

enum class InferStage {
  kContext,
  kDecode,
};

// FastPath / split-fuse 单步语义：经 IPC 下发给 Executor，决定本步是否采样、写 ring、生成 draft。
enum class ForwardStepKind : uint8_t {
  kDecode = 0,
  kPrefill,          // 普通或末块 prefill（需采样 + MTP verify）
  kPrefillChunkMid,  // split-fuse 中间 chunk：仅 forward 写 KV，不采样不回传 token
};

inline bool IsPrefillChunkMidStep(ForwardStepKind kind) { return kind == ForwardStepKind::kPrefillChunkMid; }

// build-ahead 占位 sentinel：在 step N 采样完成前提前 launch step N+1 时，尚未知的末位 token 用此值占位。
// 非 MTP 路径仍由原 ReplaceLastTokenKernel 回填；MTP 串行路径在 Executor CPU 侧修补 forwarding_tokens。
// Engine（InferRequest）与 Executor（ModelBuilder）共用此常量。
inline constexpr int kFastPathPlaceholderTokenId = -1;

// MTP 串行 CPU repair host ring 每 req 字段数: (req_id, token, accepted_draft_count, forward_draft_count)
//
// 布局示意：
//   ring = [count,
//           req_id_0, bonus_0, accepted_n_0, forward_draft_n_0,
//           req_id_1, bonus_1, ... ]
// 载荷字段（count / accepted / forward_draft）与 ring 元素同为 int64_t，读写无转换；
// 容器下标（index / base / req_count）仍用 size_t，与 vector::size()/[] 对齐。
inline constexpr size_t kBuildAheadRingFieldsPerReq = 4;

// 在 MTP 串行 host ring [count, (req_id, token, accepted, forward_draft)*] 中按 req_id 查找条目。
inline bool LookupBuildAheadHostRingEntry(const std::vector<int64_t>& ring, int64_t req_id, int64_t* token,
                                          int64_t* accepted, int64_t* forward_draft) {
  if (ring.empty()) {
    return false;
  }
  const int64_t count = ring[0];
  if (count <= 0) {
    return false;
  }
  for (int64_t j = 0; j < count; ++j) {
    const size_t base = 1 + kBuildAheadRingFieldsPerReq * static_cast<size_t>(j);
    if (base + kBuildAheadRingFieldsPerReq > ring.size()) {
      break;
    }
    if (ring[base] == req_id) {
      if (token != nullptr) {
        *token = ring[base + 1];
      }
      if (accepted != nullptr) {
        *accepted = ring[base + 2];
      }
      if (forward_draft != nullptr) {
        *forward_draft = ring[base + 3];
      }
      return true;
    }
  }
  return false;
}

// 预分配 host ring 并写入 count 头。
inline void InitBuildAheadHostRing(std::vector<int64_t>& ring, size_t req_count) {
  ring.assign(1 + kBuildAheadRingFieldsPerReq * req_count, 0);
  ring[0] = static_cast<int64_t>(req_count);
}

// 写入 ring 中第 index 个 req 条目（0-based，须 < count）。
inline void WriteBuildAheadHostRingEntry(std::vector<int64_t>& ring, size_t index, int64_t req_id, int64_t token,
                                         int64_t accepted, int64_t forward_draft) {
  const size_t base = 1 + kBuildAheadRingFieldsPerReq * index;
  ring[base] = req_id;
  ring[base + 1] = token;
  ring[base + 2] = accepted;
  ring[base + 3] = forward_draft;
}

}  // namespace ksana_llm
