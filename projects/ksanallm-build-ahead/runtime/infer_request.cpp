/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/infer_request.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <sstream>
#include <vector>
#include "profiler/timer.h"
#include "runtime/fast_path_controller.h"
#include "runtime/infer_stage.h"
#include "utils/memory_utils.h"
#include "utils/request.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {

void StampOnce(std::atomic<uint64_t> &slot) {
  if (slot.load(std::memory_order_relaxed) != 0) return;
  uint64_t expected = 0;
  slot.compare_exchange_strong(expected, ProfileTimer::GetCurrentTimeInUs(), std::memory_order_relaxed);
}

InferRequest::InferRequest(std::shared_ptr<Request> &request, const int index)
    : req_id(request->req_ids[index]),
      logits_custom_length(request->logits_custom_length),
      input_tokens(request->input_tokens),
      input_refit_embedding(request->input_refit_embedding),
      output_tokens(std::get<0>(request->output_group[index])),
      logprobs(std::get<1>(request->output_group[index])),
      return_cache_stat(request->return_cache_stat),
      cache_stat(request->cache_stat),
      request_target(request->request_target),
      response(request->response),
      sampling_config(request->sampling_config),
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      abort_waiter(request->abort_waiter),
      structured_generator_config(request->structured_generator_config),
      finished(request->finisheds[index]),
      aborted(request->aborted),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      kv_comm_group_key(request->kv_comm_group_key),
      pd_v2(*request),
      beam_search_group(request->beam_search_group),
      timestamp_in_us(ProfileTimer::GetCurrentTimeInUs()),
      req_ctx(request->req_ctx),
      prefill_entering_forward(request->prefill_entering_forward),
      scheduled_time_us(request->scheduled_time_us),
      first_token_time_us(request->first_token_time_us),
      finish_time_us(request->finish_time_us),
      prefix_cache_hit_tokens(request->prefix_cache_hit_tokens) {
  draft_tokens.spec = request->draft_tokens;
}

PdV2RequestContext::PdV2RequestContext(Request &request)
    : blocks_allocated_time_us(request.pd_v2_blocks_allocated_time_us),
      prefill_dispatched_time_us(request.pd_v2_prefill_dispatched_time_us),
      prefill_complete_time_us(request.pd_v2_prefill_complete_time_us),
      pf_prefix_cache_hit_tokens(request.pd_v2_pf_prefix_cache_hit_tokens),
      pf_queue_us(request.pd_v2_pf_queue_us),
      pf_compute_us(request.pd_v2_pf_compute_us),
      pf_send_us(request.pd_v2_pf_send_us) {}

InferRequest::~InferRequest() {
  KLLM_LOG_DEBUG << "req " << req_id << " destroyed.";
  // 触发 pd_v2 等模块注册的 teardown 回调，确保异常路径也能清理 connector 状态。
  // 析构函数保持 no-throw，回调异常只记录日志。
  for (auto &cb : on_destroy_callbacks_) {
    try {
      if (cb) cb();
    } catch (const std::exception &e) {
      KLLM_LOG_WARNING << "InferRequest::~InferRequest: on_destroy callback threw: " << e.what()
                       << " (req_id=" << req_id << ")";
    } catch (...) {
      KLLM_LOG_WARNING << "InferRequest::~InferRequest: on_destroy callback threw "
                       << "unknown exception (req_id=" << req_id << ")";
    }
  }
}

std::string InferRequest::PrintKVBlockIds(bool print_details) const {
  std::ostringstream ss;
  ss << ", kv_cache_blocks_size:" << kv_cache_blocks.size() << ", kv_cache_blocks: {";
  if (print_details) {
    ss << "{ ";
    for (auto blk_id : kv_cache_blocks) {
      ss << blk_id << ", ";
    }
    ss << "}, ";
  }
  ss << "}";
  return ss.str();
}

std::string InferRequest::ToString(bool print_details) const {
  std::ostringstream oss;
  oss << " req(req_id:" << req_id << ", step:" << step << ", sampling_token_num:" << sampling_token_num
      << ", kv_cached_token_num:" << kv_cached_token_num << ", prefix_cache_len:" << prefix_cache_len
      << ", input_tokens_size:" << input_tokens.size() << ", output_tokens_size:" << output_tokens.size()
      << ", forwarding_tokens_size:" << forwarding_tokens.size() << ", draft_tokens_size:" << draft_tokens.size()
      << ", accepted_tokens_size:" << accepted_tokens.size() << ", generated_token_size:" << generated_tokens.size()
      << PrintKVBlockIds(print_details) << ", finished:" << finished << ", aborted:" << aborted
      << ", finish_status:" << finish_status.ToString() << ", infer_stage:" << static_cast<int>(infer_stage) << " ) ";
  return oss.str();
}

std::ostream &operator<<(std::ostream &os, const InferRequest &req) {
  os << req.ToString();
  return os;
}

void InferRequest::Notify() {
  for (size_t i = 0; i < req_group.size(); i++) {
    if (!req_group[i]->finished) return;
  }

  if (sampling_config.num_beams > 1) {
    std::sort(beam_search_group.begin(), beam_search_group.end(),
              [](const OutputTuple &a, const OutputTuple &b) { return std::get<2>(a) > std::get<2>(b); });

    for (size_t i = 0; i < req_group.size() && i < beam_search_group.size(); i++) {
      req_group[i]->output_tokens = std::move(std::get<0>(beam_search_group[i]));
      req_group[i]->logprobs = std::move(std::get<1>(beam_search_group[i]));
    }
  }

  for (size_t i = 0; i < req_group.size(); i++) {
    req_group[i]->ClearReqGroup();
  }

  // After a notification, the corresponding request may be destructed.
  // So we return early to avoid accessing any variables referencing it.
  if (aborted) {
    if (abort_waiter) {
      abort_waiter->Notify();
      return;
    }
    // PD-V2 path may set aborted=true on a request that never had an abort_waiter
    // (only local_endpoint allocates it). Fall through so any waiter/step_waiter
    // still gets notified instead of dereferencing null.
  }
  if (waiter) {
    waiter->Notify();
    return;
  }
  if (step_waiter) {
    step_waiter->Notify();
  }
}

const std::vector<int> &InferRequest::GetInflightSequence() const { return forwarding_tokens; }

size_t InferRequest::GetInflightSequenceLen() const { return forwarding_tokens.size(); }

size_t InferRequest::GetInflightQueryLen() const { return forwarding_tokens.size() - kv_cached_token_num; }

size_t InferRequest::GetInflightSamplingTokenNum() const { return sampling_token_num; }

size_t InferRequest::GetPlanningSequenceLen() const {
  if (planning_workload_.prefill_token_num > 0) {
    return planning_workload_.prefill_start_offset + planning_workload_.GetTokenNum();
  }
  // Decode：forwarding 已含本步待 forward 的 placeholder 块时，
  // 勿再叠加 planning_workload_，否则调度按双倍长度申请 FA/SWA block（高并发下 SWA 池耗尽）。
  const size_t fwd_len = forwarding_tokens.size();
  const size_t planning_len = planning_workload_.GetTokenNum();
  if (planning_len == 0) {
    return fwd_len;
  }
  if (!pending_placeholder_positions_.empty()) {
    const size_t blk_start = pending_placeholder_positions_.back();
    const size_t blk_draft = pending_placeholder_draft_nums_.back();
    const size_t blk_end = blk_start + 1 + blk_draft;
    if (blk_end == fwd_len && planning_len <= 1 + blk_draft) {
      return fwd_len;
    }
  }
  return fwd_len + planning_len;
}

size_t InferRequest::GetPlanningQueryLen() const { return planning_workload_.GetTokenNum(); }

size_t InferRequest::GetPlanningSamplingTokenNum() const { return planning_workload_.sampling_token_num; }

void InferRequest::SetKvCachedTokenNum(size_t num) {
  kv_cached_token_num = num;
  prefix_cache_len = num;
}

void InferRequest::SetCacheHitStatus(size_t shared_token_num, bool is_first_prefill_step) {
  if (!return_cache_stat || infer_stage != InferStage::kContext || !is_first_prefill_step) {
    return;
  }

  cache_stat.clear();
  cache_stat.emplace_back(shared_token_num, shared_token_num);

  // return if not flexible cached
  if (flexible_cached_copy_tasks.empty()) {
    return;
  }

  std::set<int> flexible_cached_token_idx;
  for (const auto &task : flexible_cached_copy_tasks) {
    flexible_cached_token_idx.insert(task.dst_token_idx_);
  }

  int start = *flexible_cached_token_idx.begin();
  int prev = start;
  for (auto it = std::next(flexible_cached_token_idx.begin()); it != flexible_cached_token_idx.end(); ++it) {
    int curr = *it;
    if (curr != prev + 1) {
      cache_stat.emplace_back(start, prev + 1);
      start = curr;
    }
    prev = curr;
  }
  cache_stat.emplace_back(start, prev + 1);
}

void InferRequest::NotifyStep() {
  if (sampling_config.num_beams > 1) {
    int output_tokens_len = -1;
    for (size_t i = 0; i < req_group.size(); i++) {
      if (req_group[i]->finished) continue;
      output_tokens_len = output_tokens_len == -1 ? req_group[i]->output_tokens.size() : output_tokens_len;
      if (req_group[i]->output_tokens.size() != (size_t)output_tokens_len) return;
    }
  }

  StampOnce(first_token_time_us);

  if (step_waiter) {
    step_waiter->Notify();
  }
}

std::vector<int> InferRequest::GetKVOccupiedDevices() {
  std::vector<int> kv_occupied_devices;
  kv_occupied_devices = cache_manager->GetBlockAllocatorGroup()->GetBlockAllocatorDevices();
  KLLM_LOG_DEBUG << "req_id: " << req_id << ", kv_occupied_devices: " << Vector2Str(kv_occupied_devices) << ".";
  return kv_occupied_devices;
}

bool InferRequest::TryRetireBackfillNextPlaceholderBlock() {
  // 队首块 main 仍为占位符且仍有后继 inflight：该块属于下一在飞步（步 k），
  // 当前 retire 的是步 k-1（首步 decode 常无 placeholder）。仅回填 main/draft，不 pop、不裁剪。
  //
  // 时序示意（depth=2）：
  //   forwarding: [.. | PH_main | PH_d0 | PH_d1 | ...]   ← 队首仍是 -1
  //   inflight:   [step k-1 (retiring), step k]
  //   本函数：用 step k-1 的 generated/draft 回填队首块，队列不动。
  const size_t blk_start = pending_placeholder_positions_.front();
  const size_t blk_draft = pending_placeholder_draft_nums_.front();
  if (!(inflight_count_ > 1 && blk_start < forwarding_tokens.size() &&
        forwarding_tokens[blk_start] == kFastPathPlaceholderTokenId)) {
    return false;
  }

  infer_stage = InferStage::kDecode;
  output_mutex.lock();

  KLLM_CHECK_WITH_INFO(blk_start + 1 + blk_draft <= forwarding_tokens.size(),
                       "placeholder block out of forwarding range (inflight>1 backfill)" + ScheduleStateToStr());

  if (!generated_tokens.empty()) {
    forwarding_tokens[blk_start] = generated_tokens[0];
    const std::vector<int> mtp_drafts = draft_tokens.GetDraftTokens();
    for (size_t i = 0; i < blk_draft && i < mtp_drafts.size(); ++i) {
      forwarding_tokens[blk_start + 1 + i] = mtp_drafts[i];
    }
  }
  output_tokens.insert(output_tokens.end(), generated_tokens.begin(), generated_tokens.end());

  if (!output_tokens.empty() && std::find(sampling_config.stop_token_ids.begin(), sampling_config.stop_token_ids.end(),
                                          output_tokens.back()) != sampling_config.stop_token_ids.end()) {
    is_eos_generated_ = true;
  }
  output_mutex.unlock();

  remaining_workload_.generated_token_num = generated_tokens.size();
  remaining_workload_.draft_token_num = draft_tokens.size();
  planning_workload_.Reset();
  planning_workload_.generated_token_num = generated_tokens.size();
  planning_workload_.draft_token_num = draft_tokens.size();

  generated_tokens.clear();
  return true;
}

void InferRequest::RetireOwnPlaceholderBlockAndShift() {
  // 本步（步 k）retire：
  //   (a) 裁剪本块 [main|draft×M] 的被拒 draft（保留 K 个 accepted）；
  //   (b) 用本步输出回填下一在飞块（main=bonus_k、draft=mtp_k）；
  //   (c) 把其后块位置按裁剪量平移。
  //
  // 例：M=2, accepted=1 → 裁掉 1 个 draft 槽，后续 pending 下标各减 1。
  const size_t blk_start = pending_placeholder_positions_.front();
  const size_t blk_draft = pending_placeholder_draft_nums_.front();

  infer_stage = InferStage::kDecode;
  output_mutex.lock();

  pending_placeholder_positions_.pop_front();
  pending_placeholder_draft_nums_.pop_front();

  const size_t accepted_num = accepted_tokens.size();
  KLLM_CHECK_WITH_INFO(accepted_num <= blk_draft, "accepted_tokens > reserved draft slots" + ScheduleStateToStr());
  KLLM_CHECK_WITH_INFO(blk_start + 1 + blk_draft <= forwarding_tokens.size(),
                       "placeholder block out of forwarding range" + ScheduleStateToStr());

  for (size_t i = 0; i < accepted_num; ++i) {
    forwarding_tokens[blk_start + 1 + i] = accepted_tokens[i];
  }
  const size_t trimmed = blk_draft - accepted_num;
  if (trimmed > 0) {
    forwarding_tokens.erase(forwarding_tokens.begin() + static_cast<ptrdiff_t>(blk_start + 1 + accepted_num),
                            forwarding_tokens.begin() + static_cast<ptrdiff_t>(blk_start + 1 + blk_draft));
    for (auto &p : pending_placeholder_positions_) {
      if (p > blk_start) {
        p -= trimmed;
      }
    }
  }

  output_tokens.insert(output_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
  output_tokens.insert(output_tokens.end(), generated_tokens.begin(), generated_tokens.end());
  accepted_tokens.clear();

  if (!pending_placeholder_positions_.empty() && !generated_tokens.empty()) {
    const size_t nxt_start = pending_placeholder_positions_.front();
    const size_t nxt_draft = pending_placeholder_draft_nums_.front();
    KLLM_CHECK_WITH_INFO(nxt_start < forwarding_tokens.size(),
                         "next placeholder block out of range" + ScheduleStateToStr());
    forwarding_tokens[nxt_start] = generated_tokens[0];
    const std::vector<int> mtp_drafts = draft_tokens.GetDraftTokens();
    for (size_t i = 0; i < nxt_draft && i < mtp_drafts.size(); ++i) {
      forwarding_tokens[nxt_start + 1 + i] = mtp_drafts[i];
    }
  }

  if (!output_tokens.empty() && std::find(sampling_config.stop_token_ids.begin(), sampling_config.stop_token_ids.end(),
                                          output_tokens.back()) != sampling_config.stop_token_ids.end()) {
    is_eos_generated_ = true;
  }
  output_mutex.unlock();

  kv_cached_token_num = blk_start + 1 + accepted_num;
  const size_t uncached_tail = forwarding_tokens.size() - kv_cached_token_num;
  forwarding_tokens_draft_num = uncached_tail > 1 ? uncached_tail - 1 : 0;

  remaining_workload_.generated_token_num = generated_tokens.size();
  remaining_workload_.draft_token_num = draft_tokens.size();
  planning_workload_.Reset();
  planning_workload_.generated_token_num = generated_tokens.size();
  planning_workload_.draft_token_num = draft_tokens.size();

  // bonus_k 已写入下一块 main，须 clear，避免下次 LaunchPlanningTask 重复 append。
  generated_tokens.clear();
}

void InferRequest::ReserveBuildAheadPlaceholderBlock(ScheduleTask &new_inflight) {
  // 预留占位块 [main | draft×M]，M=本步计划 draft 数（MTP+PTP）。
  // launch 时真实值未知（main=上一步 bonus、draft=上一步 mtp），全部用 -1 占位；
  // 步 k-1 retire 回填本块，步 k retire 裁剪被拒 draft。
  // 预留整块使 GetPlanningSequenceLen 计入 M，调度据此分配足额 KV block。
  const size_t reserve_draft = new_inflight.workload.draft_token_num;
  const size_t block_start = forwarding_tokens.size();
  forwarding_tokens.push_back(kFastPathPlaceholderTokenId);  // main 占位
  for (size_t i = 0; i < reserve_draft; ++i) {
    forwarding_tokens.push_back(kFastPathPlaceholderTokenId);  // draft 占位
  }
  pending_placeholder_positions_.push_back(block_start);
  pending_placeholder_draft_nums_.push_back(reserve_draft);
  new_inflight.workload.generated_token_num = 1;
  new_inflight.workload.draft_token_num = reserve_draft;
  new_inflight.workload.sampling_token_num = kStepGenerateTokenNum + reserve_draft;
}

void InferRequest::AppendDecodeTokensWithoutPlaceholder(ScheduleTask &new_inflight, size_t resource_ready_token_num,
                                                        const std::vector<int> &merged_draft_tokens) {
  // 慢路径 / fast path 防御分支：append 真实 generated_tokens + speculative draft。
  const size_t effective_generated_num = generated_tokens.size();
  forwarding_tokens.insert(forwarding_tokens.end(), generated_tokens.begin(), generated_tokens.end());
  resource_ready_token_num -= effective_generated_num;
  const size_t draft_token_num = std::min(resource_ready_token_num, merged_draft_tokens.size());
  forwarding_tokens.insert(forwarding_tokens.end(), merged_draft_tokens.begin(),
                           merged_draft_tokens.begin() + draft_token_num);
  if (draft_token_num < merged_draft_tokens.size()) {
    draft_tokens.TruncDraft(draft_token_num);
  }
  new_inflight.workload.generated_token_num = effective_generated_num;
  new_inflight.workload.draft_token_num = draft_token_num;
  new_inflight.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
}

void InferRequest::ResetPrefillingTokens() {
  // Init / SyncRecompute 共用：必须丢弃上一轮 forward 在途状态，否则 recompute 后 forwarding_tokens、
  // placeholder 队列与新一轮 prefill 叠加，会出现重复 token 并可能触发 GPU MMU fault。
  forwarding_tokens.clear();
  generated_tokens.clear();
  accepted_tokens.clear();
  draft_tokens.clear();
  pending_placeholder_positions_.clear();
  pending_placeholder_draft_nums_.clear();
  forward_step_kind = ForwardStepKind::kPrefill;
  planning_task_.Reset();
  for (size_t i = 0; i < inflight_count_; ++i) {
    inflight_tasks_[i].Reset();
  }
  inflight_count_ = 0;
  is_eos_generated_ = false;
  forwarding_tokens_draft_num = 0;

  infer_stage = InferStage::kContext;
  prefilling_tokens_ = output_tokens;
  kv_cached_token_num = 0;
  step = 0;
  suggested_draft_num = 0;
  prefix_cache_len = 0;
  remaining_workload_.Reset();
  remaining_workload_.prefill_token_num = prefilling_tokens_.size();

  const size_t draft_token_num = draft_tokens.size();
  // Assumption: if logits_custom_length > 0, then sampling_token_num = logits_custom_length
  //             because user asked to generate logits for a specific number of tokens
  //             In runtime implementation, sampling_token_num is used to determine the number of logits to generate
  remaining_workload_.sampling_token_num = std::max(kStepGenerateTokenNum + draft_token_num, logits_custom_length);

  planning_workload_ = remaining_workload_;
  sampling_token_num = planning_workload_.sampling_token_num;
}

void InferRequest::AppendCachedTokens(const std::vector<int> &tokens) {
  std::lock_guard<std::mutex> guard(output_mutex);
  KLLM_LOG_DEBUG << "InferRequest::AppendCachedTokens: req_id=" << req_id << ", appending " << tokens.size()
                 << " tokens"
                 << ", output_tokens_before=" << output_tokens.size()
                 << ", prefilling_tokens_before=" << prefilling_tokens_.size();
  for (int token : tokens) {
    output_tokens.push_back(token);
    prefilling_tokens_.push_back(token);
  }
  KLLM_LOG_DEBUG << "InferRequest::AppendCachedTokens: req_id=" << req_id
                 << ", output_tokens_after=" << output_tokens.size()
                 << ", prefilling_tokens_after=" << prefilling_tokens_.size();
}

void InferRequest::SetInflightTaskGenResultEstimation(size_t generated_token_num, size_t draft_token_num) {
  KLLM_CHECK(inflight_count_ > 0);
  planning_workload_.generated_token_num = generated_token_num;
  planning_workload_.draft_token_num = draft_token_num;
  remaining_workload_.generated_token_num = generated_token_num;
  remaining_workload_.draft_token_num = draft_token_num;
}

void InferRequest::SetRemainingWorkload(const ScheduleTaskWorkload &workload) { remaining_workload_ = workload; }

void InferRequest::SetPlanningWorkload(const ScheduleTaskWorkload &workload) { planning_workload_ = workload; }

void InferRequest::SetPlanningTask() {
  KLLM_CHECK(planning_task_.IsEmpty());
  planning_task_.workload = planning_workload_;

  KLLM_CHECK(remaining_workload_.prefill_token_num >= planning_workload_.prefill_token_num);
  remaining_workload_.prefill_token_num -= planning_workload_.prefill_token_num;
  remaining_workload_.prefill_start_offset += planning_workload_.prefill_token_num;
  remaining_workload_.generated_token_num = 0;
  remaining_workload_.draft_token_num = 0;
  KLLM_CHECK((remaining_workload_.prefill_token_num + remaining_workload_.prefill_start_offset) ==
             prefilling_tokens_.size());
  planning_workload_.Reset();
}

void InferRequest::UpdateAfterInflightTaskFinished() {
  KLLM_CHECK(inflight_count_ > 0);
  const ScheduleTask &oldest_inflight = inflight_tasks_[0];

  // fast path 深度 2 下, step N 产出 EOS 后仍可能已 launch step N+1 (占位步).
  // 丢弃该多余 step 的 token, 并同步 pop placeholder 队列, 避免 inflight 与 placeholder 长度失配.
  // slow path 在 EOS 后 inflight 立即归零, 不会进入此分支.
  if (is_eos_generated_) {
    generated_tokens.clear();
    accepted_tokens.clear();
    draft_tokens.clear();
    if (!pending_placeholder_positions_.empty()) {
      pending_placeholder_positions_.pop_front();
      pending_placeholder_draft_nums_.pop_front();
    }
    return;
  }

  // If task is prefill task and not the last step, drop draft_tokens.
  // generated_tokens should be empty.
  if (IsSplitPrefillStep()) {
    generated_tokens.clear();
    draft_tokens.clear();
  } else if (!pending_placeholder_positions_.empty()) {
    // build-ahead 占位块 retire（depth=2）。慢路径 pending 为空，不会进入。
    if (TryRetireBackfillNextPlaceholderBlock()) {
      return;
    }
    RetireOwnPlaceholderBlockAndShift();
    return;
  } else {
    infer_stage = InferStage::kDecode;

    output_mutex.lock();
    if (oldest_inflight.workload.draft_token_num > 0) {
      // replace draft tokens with accepted tokens.
      forwarding_tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num + accepted_tokens.size());
      output_tokens.insert(output_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
      // Accumulate MTP/draft acceptance stats for access-log hit-rate.
      mtp_draft_token_num_ += forwarding_tokens_draft_num;
      mtp_hit_token_num_ += accepted_tokens.size();
      if (req_ctx) {
        (*req_ctx)["mtp_draft_tokens"] = std::to_string(mtp_draft_token_num_);
        (*req_ctx)["mtp_hit_tokens"] = std::to_string(mtp_hit_token_num_);
      }
      accepted_tokens.clear();
    }

    // append new tokens to output_tokens
    output_tokens.insert(output_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    // GenerationController makes sure eos only appears at the end of accepted_tokens + generated_tokens
    if (std::find(sampling_config.stop_token_ids.begin(), sampling_config.stop_token_ids.end(), output_tokens.back()) !=
        sampling_config.stop_token_ids.end()) {
      KLLM_LOG_DEBUG << "req " << req_id << " finished. output_tokens.size=" << output_tokens.size()
                     << ", eos token=" << output_tokens.back();
      is_eos_generated_ = true;
    }
    output_mutex.unlock();
  }
  // current token has kv_cache
  kv_cached_token_num = forwarding_tokens.size();

  // generated token and draft token are new workload to be processed
  remaining_workload_.generated_token_num = generated_tokens.size();
  remaining_workload_.draft_token_num = draft_tokens.size();

  // Adjust planning workload for scheduling
  if (remaining_workload_.prefill_token_num > 0 &&
      (remaining_workload_.generated_token_num > 0 || remaining_workload_.draft_token_num > 0)) {
    KLLM_LOG_SCHEDULER << ScheduleStateToStr();
    assert(false);
  }

  planning_workload_.Reset();
  if (remaining_workload_.prefill_token_num > 0) {
    planning_workload_.prefill_token_num = remaining_workload_.prefill_token_num;
  } else {
    planning_workload_.generated_token_num = generated_tokens.size();
    planning_workload_.draft_token_num = draft_tokens.size();
  }
}

void InferRequest::ResetInflightTask() {
  // 队首 oldest inflight retire 后左滑队列; slow path (inflight_count_=1) 语义与原单槽 inflight_task_ 一致.
  if (inflight_count_ == 0) {
    return;
  }
  for (size_t i = 0; i + 1 < inflight_count_; ++i) {
    inflight_tasks_[i] = inflight_tasks_[i + 1];
  }
  inflight_tasks_[inflight_count_ - 1].Reset();
  --inflight_count_;
  // pending_placeholder_positions_ 在 UpdateAfterInflightTaskFinished 中随 oldest retire 一并 pop, 此处不处理.
}

void InferRequest::LaunchPlanningTask(bool use_placeholder_for_last_token) {
  KLLM_CHECK_WITH_INFO(!planning_task_.IsEmpty(), "No planning_task" + ScheduleStateToStr());
  // inflight 容量由 BatchScheduler 的 max_depth 约束 (slow=1, fast=2).
  // placeholder 仅当已有在飞 step: 其真实末位 token 尚未经 IPC 回到 Engine.
  KLLM_CHECK_WITH_INFO(inflight_count_ < kMaxInflightDepth, "Inflight full" + ScheduleStateToStr());
  if (use_placeholder_for_last_token) {
    KLLM_CHECK_WITH_INFO(inflight_count_ > 0, "Placeholder launch requires existing inflight" + ScheduleStateToStr());
  }
  assert(planning_task_.workload.GetTokenNum() > 0);

  // 选取要写入的 inflight slot.
  ScheduleTask &new_inflight = inflight_tasks_[inflight_count_];
  new_inflight = planning_task_;
  planning_task_.Reset();
  // pd_v2: when kv_from_remote == true the peer Prefill RDMA-wrote K/V
  // for the prompt and PdV2DecodeHook::OnPrefillComplete pre-populated
  //   - forwarding_tokens = input_tokens         (size N = prompt len)
  //   - generated_tokens  = [Paris]              (just-sampled token, fed
  //                                               below at line 342 into
  //                                               forwarding_tokens for the
  //                                               first decode-step query)
  //   - kv_cached_token_num = N                  (positions 0..N-1 cached)
  //   - infer_stage = kDecode
  // The default reset `kv_cached_token_num = forwarding_tokens.size()`
  // would clobber it back to N (no-op on first step where the two are
  // already equal — but defensive: subsequent decode steps' line 285
  // bumps kv_cached, and we want this branch to be a coherent inverse
  // for the non-reset case anyway). Skip the reset on pd_v2 reqs to
  // make the contract explicit and to keep the branch obviously safe
  // even if the hook's pre-populated state diverges by 1 from the
  // forwarding_tokens length.
  if (!pd_v2.kv_from_remote) {
    SetKvCachedTokenNum(forwarding_tokens.size());
  }
  if (new_inflight.workload.prefill_token_num > 0) {
    size_t forwarded_token_num = forwarding_tokens.size();
    if (forwarded_token_num == new_inflight.workload.prefill_start_offset) {
      forwarding_tokens.insert(
          forwarding_tokens.end(), prefilling_tokens_.begin() + forwarded_token_num,
          prefilling_tokens_.begin() + forwarded_token_num + new_inflight.workload.prefill_token_num);
    } else {
      // Prefix cache hit
      forwarding_tokens.assign(prefilling_tokens_.begin(), prefilling_tokens_.begin() +
                                                               new_inflight.workload.prefill_start_offset +
                                                               new_inflight.workload.prefill_token_num);
      SetKvCachedTokenNum(new_inflight.workload.prefill_start_offset);
    }

    // 末步 prefill 追加 draft; IsSplitPrefillStep 读 oldest inflight (slow path 下即本次新 push 的 task).
    // build-ahead 补充：深度 2 下 oldest 可能是仍在飞的上一 chunk，不能再用 IsSplitPrefillStep()
    // 判定末块（会把真正末块误判成中间步）。改用本次 launch 的 workload 自身判定。
    const bool is_last_prefill_chunk =
        new_inflight.workload.prefill_token_num + new_inflight.workload.prefill_start_offset >=
        prefilling_tokens_.size();
    if (is_last_prefill_chunk) {
      // 新增：告知 Executor 本步需采样（末块 prefill）。
      forward_step_kind = ForwardStepKind::kPrefill;
      const std::vector<int> &all_draft_tokens = draft_tokens.GetDraftTokens();
      const size_t draft_token_num = all_draft_tokens.size();
      forwarding_tokens.insert(forwarding_tokens.end(), all_draft_tokens.begin(), all_draft_tokens.end());
      new_inflight.workload.draft_token_num = draft_token_num;
      new_inflight.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
    } else {
      // 新增：split-fuse 中间 chunk → 仅写 KV，不采样；Executor 据此跳过采样/draft/ring。
      forward_step_kind = ForwardStepKind::kPrefillChunkMid;
      new_inflight.workload.sampling_token_num = 0;
    }
  } else {
    auto merged_draft_tokens = draft_tokens.GetDraftTokens();
    size_t resource_ready_token_num = new_inflight.workload.generated_token_num + new_inflight.workload.draft_token_num;
    KLLM_CHECK(generated_tokens.size() <= resource_ready_token_num);

    // decode 末位 token 语义:
    //   slow path — generated_tokens 已由上一步 IPC 回流, 直接 append;
    //   fast path — 上一步采样结果尚在 device/Executor, Engine 用 placeholder 块占位, 由 repair 路径回填。
    const bool fast_path_placeholder =
        use_placeholder_for_last_token && FastPathController::GetInstance().IsEnabledAtStartup();

    if (fast_path_placeholder && generated_tokens.empty()) {
      ReserveBuildAheadPlaceholderBlock(new_inflight);
    } else {
      if (use_placeholder_for_last_token) {
        KLLM_CHECK_WITH_INFO(
            generated_tokens.size() <= 1,
            "Fast path placeholder currently only supports kStepGenerateTokenNum=1, generated_tokens>1" +
                ScheduleStateToStr());
      }
      AppendDecodeTokensWithoutPlaceholder(new_inflight, resource_ready_token_num, merged_draft_tokens);
    }
    forward_step_kind = ForwardStepKind::kDecode;
  }
  sampling_token_num = new_inflight.workload.sampling_token_num;
  forwarding_tokens_draft_num = new_inflight.workload.draft_token_num;
  ++inflight_count_;
  KLLM_LOG_SCHEDULER << ScheduleStateToStr();
  assert(new_inflight.workload.GetTokenNum() > 0);
  assert(forwarding_tokens.size() > kv_cached_token_num);
}

std::string InferRequest::ScheduleStateToStr() const {
  std::stringstream ss;
  ss << " schedule_state={ req_id=" << req_id << ", is_stopped=" << is_stopped_
     << ", inflight_count_=" << inflight_count_ << ", inflight_tasks_[0]=" << inflight_tasks_[0].workload.ToString()
     << ", inflight_tasks_[1]=" << inflight_tasks_[1].workload.ToString()
     << ", planning_task_=" << planning_task_.workload.ToString()
     << ", remaining_workload_=" << remaining_workload_.ToString()
     << ", planning_workload_= " << planning_workload_.ToString()
     << ", forwarding_tokens.size=" << forwarding_tokens.size() << ", generated_tokens=" << Vector2Str(generated_tokens)
     << ", accepted_tokens=" << Vector2Str(accepted_tokens)
     << ", draft_tokens=" << Vector2Str(draft_tokens.GetDraftTokens())
     << ", output_tokens.size=" << output_tokens.size() << ", is_eos_generated=" << is_eos_generated_;

  // Print request context headers
  if (req_ctx) {
    auto it_ip = req_ctx->find("x-remote-ip");
    auto it_group = req_ctx->find("kv-comm-group-key");
    auto it_reqid = req_ctx->find("request-id");
    auto it_trace = req_ctx->find("traceparent");
    ss << ", x_remote_ip=" << (it_ip != req_ctx->end() ? it_ip->second : "-")
       << ", kv_comm_group_key=" << (it_group != req_ctx->end() ? it_group->second : "-")
       << ", request_id=" << (it_reqid != req_ctx->end() ? it_reqid->second : "-")
       << ", traceparent=" << (it_trace != req_ctx->end() ? it_trace->second : "-");
  }
  ss << " }";
  return ss.str();
}

}  // namespace ksana_llm
