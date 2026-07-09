/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "runtime/fast_path_controller.h"
#include "runtime/model_input.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelBuilder {
 public:
  explicit ModelBuilder(std::shared_ptr<Context> context);

  Status BuildModelInput(std::shared_ptr<ScheduleOutput> schedule_output, std::shared_ptr<NewModelInput> model_input,
                         const RunMode run_mode);

  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);

  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager);

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);

  // 按 Attention 计算路径重排 batch：同 dp group 内 multi-token 在前、single-token 在后，
  // 以便连续空间加速。慢路径与 build-ahead 使用不同判定（见下方两个比较器）。
  template <typename T>
  void ReorderInferRequests(std::vector<T> &reqs) {
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      ReorderInferRequestsBuildAhead(reqs);
    } else {
      ReorderInferRequestsSlowPath(reqs);
    }
    ResetLogitsOffsetsAfterReorder(reqs);
  }

 private:
  // 慢路径（默认）：按「本步待算 token 数」与 GetDecodeTokenNumThreshold 区分 prefill/decode。
  // 排序结果：同 dp 内 [prefill..., decode...]，再按 token 数降序。
  template <typename T>
  void ReorderInferRequestsSlowPath(std::vector<T> &reqs) {
    std::sort(reqs.begin(), reqs.end(), [this](const auto &a, const auto &b) {
      // For dp case, the order is: [group1_prefill, group1_decode, group2_prefill, group2_decode, ...]
      // NOTE: prefill may appear after decode (if they are in different dp groups)
      if (a->attn_dp_group_id != b->attn_dp_group_id) {
        return a->attn_dp_group_id < b->attn_dp_group_id;
      }

      auto get_vec_size = [](const auto &item) {
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::vector<int>>) {
          return item.size();
        } else {
          return item->size();
        }
      };

      // For non-dp case, the order is: [prefill, decode]
      const size_t a_token_num = get_vec_size(a->forwarding_tokens) - a->kv_cached_token_num;
      const size_t b_token_num = get_vec_size(b->forwarding_tokens) - b->kv_cached_token_num;

      const bool is_a_decode = a_token_num <= GetDecodeTokenNumThreshold() && a->kv_cached_token_num > 0;
      const bool is_b_decode = b_token_num <= GetDecodeTokenNumThreshold() && b->kv_cached_token_num > 0;

      if (is_a_decode == is_b_decode) {
        // Both prefill or decode, sort the infer_reqs list based on a_token_num and b_token_num,
        // i.e., the number of tokens that need to be calculated for the KV cache.
        // NOTE: a_token_num or b_token_num may be zero
        if (a_token_num != b_token_num) {
          return a_token_num > b_token_num;
        }
        if (a->kv_cached_token_num != b->kv_cached_token_num) {
          return a->kv_cached_token_num < b->kv_cached_token_num;
        }
        return a->req_id < b->req_id;
      } else {
        // One is prefill, another is decode, prefill before decode
        return !is_a_decode;
      }
    });
  }

  // build-ahead：与 ForwardRequest::GetInputIdsLength / Compressor IsPaged 对齐。
  // 占位步可能出现 kv_cached==fwd_sz → input_ids_len==0；若仍用慢路径 decode 启发式，
  // 会被排到单 token paged 之后，触发 DSv4 compressor 断言。
  //
  // 例（同 dp）：
  //   A: prefill q_len=128
  //   B: build-ahead 占位步 input_ids_len=0（!IsPaged）
  //   C: decode IsPaged (q_len=1, kv>0)
  // 期望顺序：A, B, C（所有 !IsPaged 在 IsPaged 之前）。
  template <typename T>
  void ReorderInferRequestsBuildAhead(std::vector<T> &reqs) {
    std::sort(reqs.begin(), reqs.end(), [](const auto &a, const auto &b) {
      if (a->attn_dp_group_id != b->attn_dp_group_id) {
        return a->attn_dp_group_id < b->attn_dp_group_id;
      }

      auto get_vec_size = [](const auto &item) {
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::vector<int>>) {
          return item.size();
        } else {
          return item->size();
        }
      };

      auto get_input_ids_len = [&](const auto &req) -> size_t {
        const size_t flexible_cache_len = req->flexible_cached_copy_tasks.size();
        const size_t effective_prefix = static_cast<size_t>(req->prefix_cache_len) + flexible_cache_len;
        return get_vec_size(req->forwarding_tokens) - std::max(req->kv_cached_token_num, effective_prefix);
      };

      const auto is_paged = [](size_t input_ids_len, size_t kv_cached) { return input_ids_len == 1 && kv_cached > 0; };

      const size_t a_input_ids_len = get_input_ids_len(a);
      const size_t b_input_ids_len = get_input_ids_len(b);
      const bool is_a_paged = is_paged(a_input_ids_len, a->kv_cached_token_num);
      const bool is_b_paged = is_paged(b_input_ids_len, b->kv_cached_token_num);

      if (is_a_paged != is_b_paged) {
        return !is_a_paged;
      }
      if (a_input_ids_len != b_input_ids_len) {
        return a_input_ids_len > b_input_ids_len;
      }
      if (a->kv_cached_token_num != b->kv_cached_token_num) {
        return a->kv_cached_token_num < b->kv_cached_token_num;
      }
      return a->req_id < b->req_id;
    });
  }

  // 重排后按新顺序重算 logits_offset（慢路径 / ahead 共用）。
  template <typename T>
  void ResetLogitsOffsetsAfterReorder(std::vector<T> &reqs) {
    size_t logits_offset = 0;
    for (auto &req : reqs) {
      req->logits_offset = logits_offset;
      logits_offset += req->sampling_token_num;
    }
  }

  // Build forward request, group by model name and stage.
  void BuildForwardRequests(std::vector<std::shared_ptr<ExecutorInferRequest>> &reqs,
                            std::vector<ForwardRequest> &forward_reqs);

  void BuildForwardRequestFromInferRequest(ForwardRequest &forward_req,
                                           std::shared_ptr<ExecutorInferRequest> &infer_req, uint32_t layer_num);

  // Build ATB KV cache block ids
  void BuildFlatKVCacheBlkIds(uint32_t layer_num, const std::vector<int> &device_block_ids, size_t rank_num,
                              std::vector<std::vector<int32_t>> &atb_block_ids,
                              std::shared_ptr<BlockAllocatorInterface> block_allocator);

 private:
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;
  // The SWA block allocator manager; nullptr when SWA is disabled.
  std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager_ = nullptr;
};

}  // namespace ksana_llm
