/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/model_runner.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "fmt/format.h"

#include "device/device_utils.h"
#include "executor/global_cache_put_worker.h"
#include "executor/layer_progress_tracker.h"
#include "executor/model_sampler.h"
#include "pd_v2/prefill/pd_v2_layer_event_tracker.h"
#include "profiler/profile_event.h"
#include "runtime/fast_path_controller.h"
#include "runtime/infer_stage.h"
#include "runtime/structured_generation/structured_generator_interface.h"
#include "samplers/sampler.h"
#include "utils/singleton.h"
#include "utils/waiter.h"

namespace ksana_llm {

const InferRequestSerializationSnapshot* ModelRunner::FindReqSerializationSnapshot(
    const std::shared_ptr<ScheduleOutput>& schedule_output, int64_t req_id) {
  if (schedule_output == nullptr) {
    return nullptr;
  }
  const auto it = schedule_output->req_serialization_snapshots.find(req_id);
  if (it == schedule_output->req_serialization_snapshots.end()) {
    return nullptr;
  }
  return &it->second;
}

size_t ModelRunner::ResolveLaunchPlaceholderOffset(const std::shared_ptr<ExecutorInferRequest>& req,
                                                   const InferRequestSerializationSnapshot* snap) {
  if (snap != nullptr && snap->placeholder_offset_in_snapshot != std::numeric_limits<size_t>::max()) {
    return snap->placeholder_offset_in_snapshot;
  }
  if (req->launch_placeholder_offset != std::numeric_limits<size_t>::max()) {
    return req->launch_placeholder_offset;
  }
  for (size_t i = 0; i < req->forwarding_tokens.size(); ++i) {
    if (req->forwarding_tokens[i] == kFastPathPlaceholderTokenId) {
      return i;
    }
  }
  return std::numeric_limits<size_t>::max();
}

bool ModelRunner::RepairMtpSerialBlock(std::shared_ptr<ExecutorInferRequest>& req,
                                       const std::vector<int64_t>& prev_ring,
                                       const std::unordered_map<int64_t, std::vector<int>>* accepted_map, size_t ph_off,
                                       size_t draft_slots, size_t old_main_pos, size_t old_draft_slots,
                                       const std::vector<int>* frozen_mtp_drafts) {
  int64_t token = 0;
  int64_t accepted = 0;
  const bool found = LookupBuildAheadHostRingEntry(prev_ring, req->req_id, &token, &accepted, nullptr);
  if (!found) {
    return false;
  }
  // ring 中不应出现 placeholder sentinel；若出现说明上步未产出有效 bonus，跳过 repair 保留占位块。
  if (token == kFastPathPlaceholderTokenId) {
    return false;
  }
  if (ph_off == std::numeric_limits<size_t>::max() || ph_off + draft_slots >= req->forwarding_tokens.size()) {
    return false;
  }

  const std::vector<int>* acc_vals = nullptr;
  if (accepted_map != nullptr) {
    const auto it = accepted_map->find(req->req_id);
    if (it != accepted_map->end()) {
      acc_vals = &it->second;
    }
  }
  // accepted 与 ring 同为 int64_t；与 vector::size() 比较时再落到 size_t，并拒绝负值。
  const size_t accepted_n = std::min(accepted > 0 ? static_cast<size_t>(accepted) : 0, acc_vals ? acc_vals->size() : 0);

  // 上一 verify 块信息由调用方传入（不能用 req->kv_cached_token_num/forwarding_tokens_draft_num，
  // 每步 UpdateCachedRequests 会用 Engine 的 IPC patch 覆盖，patch 里的值滞后于 Executor 本地已修补的
  // 真实块位置）。分两种情况：
  //  - old_draft_slots>0：已知上一步是真实 verify 块，accepted draft 原地覆写到上一块槽位，
  //    裁剪 [新主槎, ph_off) 之间残留的被拒 draft（此前从未清理，是 KV 位置持续漂移的根因——被拒
  //    draft 会被误当作已 cache 的正式 token，导致后续每步 position/KV 都偏移，采样发散）。
  //  - old_draft_slots==0：无历史块记录（紧接 prefill 首次修补 / 分布式快照恢复，snapshot 已是
  //    Engine 侧裁剪好的状态），维持原逻辑：accepted 插入 ph_off 之前。
  size_t main_pos;
  if (old_draft_slots > 0) {
    if (accepted_n > 0 && acc_vals != nullptr) {
      for (size_t i = 0; i < accepted_n; ++i) {
        req->forwarding_tokens[old_main_pos + 1 + i] = (*acc_vals)[i];
      }
    }
    const size_t new_main_pos = old_main_pos + 1 + accepted_n;
    if (ph_off > new_main_pos) {
      req->forwarding_tokens.erase(req->forwarding_tokens.begin() + static_cast<ptrdiff_t>(new_main_pos),
                                   req->forwarding_tokens.begin() + static_cast<ptrdiff_t>(ph_off));
    }
    main_pos = new_main_pos;
  } else {
    if (accepted_n > 0 && acc_vals != nullptr) {
      req->forwarding_tokens.insert(req->forwarding_tokens.begin() + static_cast<ptrdiff_t>(ph_off), acc_vals->begin(),
                                    acc_vals->begin() + static_cast<ptrdiff_t>(accepted_n));
    }
    main_pos = ph_off + accepted_n;
  }

  // main 槽 = 上一步 bonus（ring token）。
  const int main_tok = static_cast<int>(token);
  req->forwarding_tokens[main_pos] = main_tok;

  // M 个 draft 槽 = 上一步 MtpForward 结束后冻结的 MTP draft（与 ring 同周期），不用 live draft_tokens。
  std::vector<int> mtp_drafts;
  if (frozen_mtp_drafts != nullptr) {
    mtp_drafts = *frozen_mtp_drafts;
  } else {
    mtp_drafts = req->draft_tokens.mtp;
  }
  size_t draft_fill = 0;
  for (size_t i = 0; i < mtp_drafts.size() && draft_fill < draft_slots; ++i) {
    req->forwarding_tokens[main_pos + 1 + draft_fill] = mtp_drafts[i];
    ++draft_fill;
  }
  if (draft_fill < draft_slots) {
    req->forwarding_tokens.erase(req->forwarding_tokens.begin() + static_cast<ptrdiff_t>(main_pos + 1 + draft_fill),
                                 req->forwarding_tokens.begin() + static_cast<ptrdiff_t>(main_pos + 1 + draft_slots));
  }

  // kv 指向 main（verify 输入起点），sampling=1+M，draft_num=M。
  req->kv_cached_token_num = main_pos;
  req->prefix_cache_len = static_cast<int>(main_pos);
  req->forwarding_tokens_draft_num = draft_fill;
  req->sampling_token_num = kStepGenerateTokenNum + draft_fill;
  return true;
}

ModelRunner::ModelRunner(std::shared_ptr<Context> context) : mtp_step_num_(context->global->GetMtpStepNum()) {
  context_ = context;

  schedule_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<ScheduleOutput>>>();
  model_input_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelInput>>>();
  model_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<NewModelOutput>>>();
  sampling_output_queue_ = std::make_shared<BlockingQueue<std::shared_ptr<SamplingOutput>>>();
}

void ModelRunner::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

void ModelRunner::SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime) {
  executor_runtime_ = executor_runtime;
}

void ModelRunner::SetModelInstance(std::shared_ptr<ModelInstance> model_instance) {
  model_instance_ = model_instance;
  executor_runtime_->SetModelInstance(model_instance_);
}

void ModelRunner::SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight) {
  model_weight_ = model_weight;
  executor_runtime_->SetModelWeight(model_weight_);
}

void ModelRunner::SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager) {
  block_allocator_manager_ = block_allocator_manager;
  executor_runtime_->SetBlockAllocatorManager(block_allocator_manager_);
}

void ModelRunner::SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager) {
  executor_runtime_->SetSwaBlockAllocatorManager(swa_block_allocator_manager);
}

void ModelRunner::SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool) {
  hidden_unit_buffer_pool_ = hidden_unit_buffer_pool;
  executor_runtime_->SetHiddenUnitBufferPool(hidden_unit_buffer_pool);
}

void ModelRunner::SetSamplingOutputPool(std::shared_ptr<SamplingOutputPool> sampling_output_pool) {
  sampling_output_pool_ = sampling_output_pool;
}

void ModelRunner::SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller) {
  executor_runtime_->SetDraftGeneratorController(draft_generator_controller);
}

void ModelRunner::SetGlobalCachePutWorker(std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker) {
  global_cache_put_worker_ = global_cache_put_worker;
}

Status ModelRunner::Start() {
  // Store the inflight resource manager pointer in ModelInstance so that Load()
  // can register resources once the base model is available. Both fields must
  // be ready at this point: executor_runtime_ is constructed in Initialize()
  // (which always runs before Start()), and model_instance_ is set via
  // SetModelInstance() before Start() is called.
  KLLM_CHECK_WITH_INFO(executor_runtime_ != nullptr, "executor_runtime_ must be set before Start()");
  KLLM_CHECK_WITH_INFO(model_instance_ != nullptr, "model_instance_ must be set before Start()");
  model_instance_->SetInflightResourceManager(executor_runtime_->GetInflightResourceManager().get());

  // Cache raw pointer to avoid a shared_ptr copy (atomic refcount) on the hot preprocess path.
  // Safe: executor_runtime_ outlives all worker threads started below.
  inflight_resource_mgr_ = executor_runtime_->GetInflightResourceManager().get();

  // MTP 串行 + Engine pipeline depth=2：限制 Preprocess 入队深度，与全局在途 schedule 数对齐。
  if (IsMtpSerialFastPath()) {
    constexpr uint32_t kMtpSerialModelInputQueueMaxDepth = 2;
    model_input_queue_ =
        std::make_shared<BlockingQueue<std::shared_ptr<NewModelInput>>>(kMtpSerialModelInputQueueMaxDepth);
  }

  preprocess_thread_ = std::thread(&ModelRunner::HandlePreprocess, this);
  execute_thread_ = std::thread(&ModelRunner::HandleExecute, this);

  if (context_->global->IsSamplingNode()) {
    if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      // build-ahead: Forward/Sampling/WriteRing 在 Execute 线程串行, IPC 上报走独立线程.
      result_report_thread_ = std::thread(&ModelRunner::HandleResultReportProcess, this);
    } else {
      // 慢路径: Preprocess / Execute / Postprocess 三线程（build-ahead 关闭时的默认模型）。
      postprocess_thread_ = std::thread(&ModelRunner::HandlePostprocess, this);
    }
  }

  return Status();
}

Status ModelRunner::Stop() {
  terminated_.store(true, std::memory_order_relaxed);

  schedule_output_queue_->Stop();
  model_input_queue_->Stop();
  model_output_queue_->Stop();
  sampling_output_queue_->Stop();

  // Wake the execute thread if it is currently parked inside ExecutorRuntime::Forward()'s wait for
  // the previous MtpForward Notify(); without this the join below could hang on a Notify that
  // HandlePostprocess will never deliver after the queues have been stopped.
  if (executor_runtime_) {
    executor_runtime_->Shutdown();
  }

  if (preprocess_thread_.joinable()) {
    preprocess_thread_.join();
  }
  if (execute_thread_.joinable()) {
    execute_thread_.join();
  }
  if (postprocess_thread_.joinable()) {
    postprocess_thread_.join();
  }
  if (result_report_thread_.joinable()) {
    result_report_thread_.join();
  }

  return Status();
}

void ModelRunner::GetHiddenShape(const std::shared_ptr<ScheduleOutput>& schedule_output, std::vector<size_t>& shape) {
  std::vector<size_t> input_ids_cpu;
  for (const auto& req : schedule_output->executor_running_reqs) {
    const size_t skip_token_num = std::max(req->kv_cached_token_num, static_cast<size_t>(req->prefix_cache_len));
    input_ids_cpu.insert(input_ids_cpu.end(), req->forwarding_tokens.begin() + skip_token_num,
                         req->forwarding_tokens.end());
  }
  shape = {input_ids_cpu.size(),
           model_instance_->GetModelConfig().hidden_units * model_instance_->GetModelConfig().mhc_config.hc_mult};
}

DataType ModelRunner::GetHiddenDataType() { return model_instance_->GetModelConfig().weight_data_type; }

Status ModelRunner::HandlePreprocess() {
  SetDevice(context_->device->GetDeviceId());

  ipc_func_wrapper_->SetScheduleOutputRecvQueue(schedule_output_queue_);

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<ScheduleOutput> schedule_output;
    {
      PROFILE_EVENT_SCOPE(preprocess_queue_get, "Exec.HandlePreprocess.QueueGet");
      schedule_output = schedule_output_queue_->Get();
    }
    if (!schedule_output) {
      KLLM_LOG_INFO << "HandlePreprocess stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(preprocess_iter, "Exec.HandlePreprocess.iter");

    const bool mtp_serial = IsMtpSerialFastPath();
    // MTP 串行 + pipeline depth=2：Preprocess 可超前入队 schedule，但不得提前改 executor 侧 req 缓存，
    // 否则下一步 Execute Forward 可能读到被后序 schedule 覆写的状态（CUDA illegal access）。
    if (!mtp_serial) {
      {
        PROFILE_EVENT_SCOPE(preprocess_update_cached, "Exec.HandlePreprocess.UpdateCachedRequests");
        executor_runtime_->UpdateCachedRequests(schedule_output);
      }
      {
        PROFILE_EVENT_SCOPE(preprocess_rebuild, "Exec.HandlePreprocess.RebuildScheduleOutput");
        executor_runtime_->RebuildScheduleOutput(schedule_output);
      }

      auto generation_controller = executor_runtime_->GetGenerationController();
      if (generation_controller) {
        PROFILE_EVENT_SCOPE(preprocess_generator, "Exec.HandlePreprocess.RestoreGenerators");
        if (!schedule_output->need_create_generator_req_ids.empty()) {
          generation_controller->CreateGenerators(schedule_output);
        }
        generation_controller->RestoreGenerators(schedule_output);
      }
    }

    if (!context_->global->IsStandalone()) {
      KLLM_CHECK_WITH_INFO(!mtp_serial,
                           "mtp_serial fast path with distributed mode is unsupported (hidden meta needs rebuild).");
      PROFILE_EVENT_SCOPE(preprocess_hidden_meta, "Exec.HandlePreprocess.HiddenMeta");
      std::vector<size_t> shape;
      GetHiddenShape(schedule_output, shape);
      DataType data_type = GetHiddenDataType();
      if (context_->global->IsDistributedWorker()) {
        hidden_unit_buffer_pool_->AddHiddenUnitRecvMeta(shape, data_type);
      }

      if (!context_->global->IsDistributedLastWorker()) {
        hidden_unit_buffer_pool_->AddHiddenUnitSendMeta(shape, data_type);
      }
    }

    if (!mtp_serial) {
      // Compressor slot alloc/free is now handled by InflightResourceManager::ProcessScheduleOutput,
      // which must be called before BuildModelInput so that new running_reqs have their slots
      // allocated and finished/recomputed reqs have their slots freed before PrepareCompressor runs.
      //
      // MTP serial fast path defers BuildModelInput to Execute (compute-serial design), so its resource
      // lifecycle must also be deferred there. Otherwise launch-ahead ScheduleOutputs would occupy
      // compressor slots before their actual compute step.
      PROFILE_EVENT_SCOPE(preprocess_inflight_resource, "Exec.HandlePreprocess.InflightResource");
      inflight_resource_mgr_->ProcessScheduleOutput(*schedule_output);
    }

    std::shared_ptr<NewModelInput> model_input = std::make_shared<NewModelInput>();
    model_input->schedule_output = schedule_output;
    model_input->run_mode = RunMode::kMain;

    if (!mtp_serial) {
      PROFILE_EVENT_SCOPE(preprocess_build_model_input, "Exec.HandlePreprocess.BuildModelInput");
      executor_runtime_->BuildModelInput(schedule_output, model_input, RunMode::kMain);
    }

    // ModelInput / ring 双 slot 乒乓翻转, 使 Preprocess 与 Execute 可重叠且互不覆写.
    // 慢路径恒用 slot 0；FastPath+MTP 串行恒用 slot 0（Execute 串行 Forward, Preprocess 可 depth=2 超前入队）。
    if (FastPathController::GetInstance().IsEnabledAtStartup() && !mtp_serial) {
      cur_ring_slot_ ^= 1u;
      model_input->slot_index = cur_ring_slot_;
    } else {
      model_input->slot_index = 0;
    }

    // 双 H2D stream 开启时, 在 Preprocess 线程完成 ParseFromRequests, 与上一步 Forward 重叠.
    model_input->gpu_input_ready = false;
    if (!mtp_serial && context_->device->IsDualH2DStreamsEnabled()) {
      Status prep_status = executor_runtime_->PrepareModelInputGpu(model_input);
      if (!prep_status.OK()) {
        KLLM_LOG_ERROR << "PrepareModelInputGpu failed: " << prep_status.ToString()
                       << " schedule_id=" << schedule_output->schedule_id;
        return prep_status;
      }
    }

    model_input_queue_->Put(model_input);
  }
  return Status();
}

Status ModelRunner::HandleExecute() {
  SetDevice(context_->device->GetDeviceId());

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<NewModelInput> model_input;
    {
      PROFILE_EVENT_SCOPE(execute_input_get, "Exec.HandleExecute.InputQueueGet");
      model_input = model_input_queue_->Get();
    }
    if (!model_input) {
      KLLM_LOG_INFO << "HandleExecute stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(execute_iter, "Exec.HandleExecute.iter");

    const bool mtp_serial = IsMtpSerialFastPath();
    if (mtp_serial) {
      {
        PROFILE_EVENT_SCOPE(execute_update_cached, "Exec.HandleExecute.UpdateCachedRequests");
        executor_runtime_->UpdateCachedRequests(model_input->schedule_output);
      }
      {
        PROFILE_EVENT_SCOPE(execute_rebuild, "Exec.HandleExecute.RebuildScheduleOutput");
        executor_runtime_->RebuildScheduleOutput(model_input->schedule_output);
      }
      auto generation_controller = executor_runtime_->GetGenerationController();
      if (generation_controller) {
        PROFILE_EVENT_SCOPE(execute_generator, "Exec.HandleExecute.RestoreGenerators");
        if (!model_input->schedule_output->need_create_generator_req_ids.empty()) {
          generation_controller->CreateGenerators(model_input->schedule_output);
        }
        generation_controller->RestoreGenerators(model_input->schedule_output);
      }
      {
        PROFILE_EVENT_SCOPE(execute_inflight_resource, "Exec.HandleExecute.InflightResource");
        inflight_resource_mgr_->ProcessScheduleOutput(*model_input->schedule_output, /*free_inactive=*/true);
      }
      ClearMtpSerialStateForLifecycle(model_input->schedule_output);
      const std::vector<int64_t>& prev_ring = mtp_serial_host_ring_;
      ApplyCpuBuildAheadRepairFromRing(model_input->schedule_output, prev_ring);
      {
        PROFILE_EVENT_SCOPE(execute_build_model_input, "Exec.HandleExecute.BuildModelInput");
        Status build_status =
            executor_runtime_->BuildModelInput(model_input->schedule_output, model_input, RunMode::kMain);
        if (!build_status.OK()) {
          return build_status;
        }
      }
      model_input->prev_ring_dev = nullptr;
      model_input->gpu_input_ready = false;
    } else if (FastPathController::GetInstance().IsEnabledAtStartup()) {
      const size_t prev_slot = model_input->slot_index ^ 1u;
      auto model_sampler = executor_runtime_->GetModelSampler();
      KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                           "build_ahead_enabled but ModelSampler is null in HandleExecute (config inconsistency).");
      auto sampler = model_sampler->GetSampler();
      KLLM_CHECK_WITH_INFO(sampler != nullptr,
                           "build_ahead_enabled but Sampler is null in HandleExecute (config inconsistency).");
      int64_t* prev_ring = sampler->GetRingSlot(prev_slot);
      KLLM_CHECK_WITH_INFO(
          prev_ring != nullptr,
          fmt::format("build_ahead_enabled but Sampler ring slot {} not allocated (config inconsistency).", prev_slot));
      model_input->prev_ring_dev = prev_ring;
    }

    std::shared_ptr<NewModelOutput> model_output = std::make_shared<NewModelOutput>();

    HiddenUnitDeviceBuffer* hidden_unit = nullptr;
    {
      PROFILE_EVENT_SCOPE(execute_get_buffer, "Exec.HandleExecute.GetHiddenBuffer");
      hidden_unit = hidden_unit_buffer_pool_->GetBuffer();
    }
    GetHiddenShape(model_input->schedule_output, hidden_unit->tensor.shape);
    hidden_unit->tensor.dtype = GetHiddenDataType();

    // pd_v2: open a fresh epoch slot for this batch BEFORE Forward kicks off
    // any RecordLayerProgress. The epoch_id rides on model_output to
    // HandlePostprocess so ResetState targets this batch's slot only —
    // never racing with the next batch's writes. No-op (returns 0) on
    // non-pd_v2 / non-Prefill processes.
    model_output->pd_v2_epoch_id = Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance()->BeginBatch();

    {
      PROFILE_EVENT_SCOPE(execute_forward_launch, "Exec.HandleExecute.ForwardLaunch");
      executor_runtime_->Forward(model_input, hidden_unit, model_output);
    }

    if (global_cache_put_worker_) {
      global_cache_put_worker_->SubmitBatchPut(model_input->schedule_output->executor_running_reqs);
    }

    if (context_->global->IsSamplingNode()) {
      model_output->hidden_unit = hidden_unit;
      if (FastPathController::GetInstance().IsEnabledAtStartup()) {
        // fast path: Forward 与 Sampling/WriteRing 同线程串行, 结果上报走独立线程.
        HandlePostprocessInline(model_output);
      } else {
        // slow path: 三线程模型, Sampling 在 HandlePostprocess.
        model_output_queue_->Put(model_output);
      }
    } else {
      {
        PROFILE_EVENT_SCOPE(execute_stream_sync, "Exec.HandleExecute.NonSamplingStreamSync");
        StreamSynchronize(context_->device->GetPrimaryComputeStream());
      }
      hidden_unit_buffer_pool_->FreeBuffer(hidden_unit);
      // Non-sampling node: model_output never reaches HandlePostprocess, so
      // we must seal the pd_v2 epoch slot here to avoid leaking it from the
      // tracker's free pool.
      Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance()->ResetState(model_output->pd_v2_epoch_id);
    }

    // TP: 非 MTP 串行 fast path 在采样卡 rank 0 WriteRing 后，所有 rank 必须参与 collective。
    // MTP 串行只使用每个 rank 本地 host ring 修补 BuildModelInput，不写/读 device ring。
    if (FastPathController::GetInstance().IsEnabledAtStartup() && !IsMtpSerialFastPath()) {
      BroadcastBuildAheadRingSlot(model_output->slot_index);
    }
  }
  return Status();
}

void ModelRunner::BroadcastBuildAheadRingSlot(size_t slot) {
  if (context_->global->GetTensorParallelSize() <= 1) {
    return;
  }
  auto model_sampler = executor_runtime_->GetModelSampler();
  KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                       "build_ahead_enabled but ModelSampler is null in BroadcastBuildAheadRingSlot.");
  auto sampler = model_sampler->GetSampler();
  KLLM_CHECK_WITH_INFO(sampler != nullptr, "build_ahead_enabled but Sampler is null in BroadcastBuildAheadRingSlot.");
  sampler->BroadcastRingSlot(slot, context_->device->GetPrimaryComputeStream());
}

// TODO(robertyuan): Change serializing struct from SamplingOutput to GenerationResult.
void SerializeGenerationResultToSamplingResult(const std::unordered_map<int64_t, GenerationResult>& generation_results,
                                               std::shared_ptr<SamplingOutput> sampling_output) {
  for (auto& [req_id, result] : generation_results) {
    result.SerializeToVector(sampling_output->sampling_result[req_id]);
  }
}

Status ModelRunner::HandlePostprocess() {
  SetDevice(context_->device->GetDeviceId());
  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<NewModelOutput> model_output;
    // build-ahead：Execute 线程经 HandlePostprocessInline 注入本步输出，不经 model_output_queue_。
    // 慢路径：独立 postprocess 线程消费 model_output_queue_，postprocess_one_shot_ 恒为 false。
    bool one_shot_iteration = false;
    if (postprocess_one_shot_) {
      model_output = std::move(postprocess_one_shot_output_);
      postprocess_one_shot_ = false;
      one_shot_iteration = true;
    } else {
      {
        PROFILE_EVENT_SCOPE(postprocess_output_get, "Exec.HandlePostprocess.OutputQueueGet");
        model_output = model_output_queue_->Get();
      }
    }
    if (!model_output) {
      KLLM_LOG_INFO << "HandlePostprocess stopped.";
      break;
    }

    PROFILE_EVENT_SCOPE(postprocess_iter, "Exec.HandlePostprocess.iter");
    // split-fuse 中间 chunk：本步只 forward 写 KV，不产出 token，跳过采样/生成态更新/draft/MTP/ring。
    // 清空可能残留的采样相关字段，避免误传给下游（finish 判定、结果上报等）。
    bool need_sampling = false;
    for (const auto& req : model_output->schedule_output->executor_running_reqs) {
      if (IsPrefillChunkMidStep(req->forward_step_kind)) {
        req->generated_tokens.clear();
        req->accepted_tokens.clear();
        req->draft_tokens.clear();
        req->sampling_result_tokens.clear();
      } else {
        need_sampling = true;
      }
    }
    std::shared_ptr<SamplingOutput> sampling_output = std::make_shared<SamplingOutput>();
    // chunk mid 步跳过 Sampling() 时仍需回填 schedule_id/schedule_output，否则 Engine 侧按 schedule_id=0
    // 查找会 FATAL（历史 bug，见 NOTES.md "schedule_id 0 not found"）。
    sampling_output->schedule_id = model_output->schedule_output->schedule_id;
    sampling_output->schedule_output = model_output->schedule_output;
    if (need_sampling) {
      PROFILE_EVENT_SCOPE(postprocess_sampling, "Exec.HandlePostprocess.Sampling");
      executor_runtime_->Sampling(model_output, sampling_output);
    }
    // build-ahead 专有：Sampling 后写 ring slot；慢路径 postprocess_enqueue_report_=false 时函数内直接返回。
    // MTP 串行路径只需要 host ring，且 token 必须是 draft verify 后的主 generated token，
    // 因此延后到 UpdateGenerationState 之后写入。
    if (need_sampling && !IsMtpSerialFastPath()) {
      WriteBuildAheadRingAfterSampling(model_output);
    }

    auto generation_controller = executor_runtime_->GetGenerationController();
    {
      PROFILE_EVENT_SCOPE(postprocess_update_gen_state, "Exec.HandlePostprocess.UpdateGenerationState");
      std::vector<std::shared_ptr<ExecutorInferRequest>> gen_state_reqs;
      for (auto& req : model_output->schedule_output->executor_running_reqs) {
        if (!IsPrefillChunkMidStep(req->forward_step_kind)) {
          gen_state_reqs.push_back(req);
        }
      }
      if (!gen_state_reqs.empty()) {
        generation_controller->UpdateGenerationState(gen_state_reqs);
      }
    }
    if (IsMtpSerialFastPath()) {
      WriteMtpSerialHostRingAfterGenerationState(model_output);
    }

    std::unordered_map<int64_t, GenerationResult> generation_results;
    std::vector<int64_t> finished_req_ids;
    const bool has_profile = model_output->schedule_output && model_output->schedule_output->enable_profile_metrics;
    for (auto& req : model_output->schedule_output->executor_running_reqs) {
      auto& result = generation_results[req->req_id];
      result.generated_tokens = req->generated_tokens;
      result.accepted_tokens = req->accepted_tokens;
      sampling_output->target_logprobs[req->req_id] = req->logprobs;
      if (req->structured_generator && req->structured_generator->IsTerminated()) {
        finished_req_ids.push_back(req->req_id);
      }
    }
    // Batch-level profile metrics: write once to SamplingOutput, not per-request.
    if (has_profile) {
      sampling_output->has_profile_metrics = true;
      sampling_output->profile_metrics = model_output->profile_metrics;
    }

    if (!finished_req_ids.empty() && generation_controller) {
      for (int64_t req_id : finished_req_ids) {
        generation_controller->RemoveGenerator(req_id, model_output->schedule_output->executor_running_reqs);
        KLLM_LOG_DEBUG << "Cleaned up generator for finished req_id: " << req_id;
      }
    }

    HiddenUnitDeviceBuffer* const hidden_unit = model_output->hidden_unit;

    {
      PROFILE_EVENT_SCOPE(postprocess_gen_draft, "Exec.HandlePostprocess.GenerateDraft");
      for (auto& req : model_output->schedule_output->executor_running_reqs) {
        if (IsPrefillChunkMidStep(req->forward_step_kind)) {
          continue;
        }
        generation_controller->GenerateDraft(req);
        // build-ahead：Execute 与下一步 Forward 重叠，需把 draft 计入 sampling_token_num 供调度/workload 估算。
        // 慢路径：Sampling 在独立线程，req 状态在入队前已定型，此处无需再改 sampling_token_num。
        if (postprocess_enqueue_report_) {
          size_t draft_token_num = req->draft_tokens.size();
          req->sampling_token_num = kStepGenerateTokenNum + draft_token_num;
        }
      }
    }

    if (context_->global->IsSamplingNode() && mtp_step_num_ > 0) {
      std::vector<std::shared_ptr<ExecutorInferRequest>> mtp_reqs;
      for (auto& req : model_output->schedule_output->executor_running_reqs) {
        if (!IsPrefillChunkMidStep(req->forward_step_kind)) {
          mtp_reqs.push_back(req);
        }
      }
      if (!mtp_reqs.empty()) {
        PROFILE_EVENT_SCOPE(postprocess_mtp_forward, "Exec.HandlePostprocess.MtpForward");
        GetHiddenShape(model_output->schedule_output, hidden_unit->tensor.shape);
        hidden_unit->tensor.dtype = GetHiddenDataType();
        if (has_profile) {
          model_output->schedule_output->step_metrics = &sampling_output->profile_metrics;
        }
        executor_runtime_->MtpForward(mtp_step_num_, mtp_reqs, hidden_unit, model_output->schedule_output);
        model_output->schedule_output->step_metrics = nullptr;
        // 冻结 MTP draft，供下一步 Execute repair 与 ring 配对使用（不得读 post-repair 的 live draft）。
        for (auto& req : mtp_reqs) {
          mtp_serial_next_mtp_drafts_[req->req_id] = req->draft_tokens.mtp;
        }
      }
    }

    // Compressor slot eviction has been moved to the start of HandleExecute (before the main
    // Forward), uniformly on both sampling and non-sampling ranks. See the comment there for the
    // rationale: doing evict before Prepare lets a single schedule_output carry finish_req_ids and
    // new running_reqs together without exhausting the slot pool. MTP on this rank only Prepare()s
    // for executor_running_reqs of the same schedule_output, which never overlap with that SO's
    // finish_req_ids, so MTP does not require a dedicated evict point here.

    {
      PROFILE_EVENT_SCOPE(postprocess_free_buffer, "Exec.HandlePostprocess.FreeHiddenBuffer");
      hidden_unit_buffer_pool_->FreeBuffer(hidden_unit);
    }
    Singleton<LayerProgressTracker>::GetInstance()->ResetState();
    // pd_v2 tracker: seal this batch's epoch slot. Force-fires any pending
    // layer callback the monitor thread hasn't observed yet, then returns
    // the slot to the free pool. epoch_id was assigned at HandleExecute
    // time via BeginBatch and rides on model_output. No-op when 0 (pd_v2
    // not active on this process).
    Singleton<pd_v2::PdV2LayerEventTracker>::GetInstance()->ResetState(model_output->pd_v2_epoch_id);

    for (auto& req : model_output->schedule_output->executor_running_reqs) {
      generation_results[req->req_id].draft_tokens = req->draft_tokens;
    }

    if (global_cache_put_worker_) {
      KLLM_LOG_DEBUG << "ModelRunner::HandlePostprocess: calling SubmitFirstTokensPut, req_count="
                     << model_output->schedule_output->executor_running_reqs.size();
      global_cache_put_worker_->SubmitFirstTokensPut(model_output->schedule_output->executor_running_reqs);
    }

    {
      PROFILE_EVENT_SCOPE(postprocess_serialize_gen, "Exec.HandlePostprocess.SerializeGenerationResult");
      SerializeGenerationResultToSamplingResult(generation_results, sampling_output);
    }

    // 结果出口：fast 入队给 HandleResultReportProcess，避免 Execute 线程阻塞 IPC；
    // slow 在 postprocess 线程、device 0 上直接上报（三线程模型下无独立 report 线程）。
    if (postprocess_enqueue_report_) {
      sampling_output_queue_->Put(sampling_output);
      postprocess_enqueue_report_ = false;
    } else if (context_->device->GetDeviceId() == 0) {
      PROFILE_EVENT_SCOPE(postprocess_report, "Exec.HandlePostprocess.Report");
      if (context_->global->IsStandalone()) {
        ipc_func_wrapper_->ReportSamplingOutput(sampling_output);
      } else if (context_->global->IsDistributedResultReporter()) {
        sampling_output_pool_->PutSendSamplingOutput(sampling_output.get());
      }
    }

    // build-ahead：Execute 线程单次调用，处理完即返回；不进入下一轮 while（慢路径无此分支）。
    if (one_shot_iteration) {
      break;
    }
  }
  return Status();
}

// 慢路径 vs FastPath+MTP 串行（本函数只出现在快路径右侧）：
//
//   慢路径 (max_depth=1, 无 host ring)          FastPath + MTP 串行 (max_depth=2, host ring)
//   ─────────────────────────────────          ────────────────────────────────────────────
//   Preprocess: Update → BuildModelInput       Preprocess: 仅 Put(schedule)（不 Update/Build）
//        │                                          │
//        ▼                                          ▼
//   Execute: Forward → Sample/Verify/MTP       Execute: Update → Repair(读 ring) → Build
//        │                                          │         → Forward → Sample/Verify/MTP
//        ▼                                          ▼
//   Postprocess: 上报结果                      本函数 WriteHostRing(bonus/accepted)
//                                              → 下一步 Execute 的 Repair 消费
//
//   Engine 侧：一步回流后再 launch 下一步       Engine 侧：步 N 未回流即可 launch 步 N+1（占位块）
//
void ModelRunner::WriteMtpSerialHostRingAfterGenerationState(const std::shared_ptr<NewModelOutput>& model_output) {
  // GenerationState 更新后写 host ring，供下一步 Execute 在 BuildModelInput 前做 CPU repair。
  //
  // 布局：[count, (req_id, bonus_token, accepted_n, forward_draft_n)*]
  // 策略：本批有 generated 的 req 覆写；未入本批的旧条目保留（勿整表清空）。
  // 例：req A 本步因 block 压力未入 batch，仍保留 A 的上一步 bonus，Recover 回 decode 后可 repair。
  if (!postprocess_enqueue_report_ || model_output == nullptr || model_output->schedule_output == nullptr) {
    return;
  }
  const auto& reqs = model_output->schedule_output->executor_running_reqs;
  // 本批产出 bonus 的 req：更新/写入 ring；未入本批的旧条目保留。
  // 这样 block 压力下「空一轮仍 decode 重入」时，仍能查到上一步 bonus 做 repair（Recover 回 decode 的高效路径）。
  std::vector<std::shared_ptr<ExecutorInferRequest>> ring_reqs;
  ring_reqs.reserve(reqs.size());
  for (const auto& req : reqs) {
    if (!req->generated_tokens.empty()) {
      ring_reqs.push_back(req);
    }
  }
  if (ring_reqs.empty()) {
    return;
  }

  // 收集本批 req_id，便于从旧 ring 里挑出「未入本批」的条目。
  std::unordered_set<int64_t> cur_ids;
  cur_ids.reserve(ring_reqs.size());
  for (const auto& req : ring_reqs) {
    cur_ids.insert(req->req_id);
  }

  std::vector<int64_t> kept;
  if (!mtp_serial_host_ring_.empty()) {
    const int64_t old_count = mtp_serial_host_ring_[0];
    for (int64_t j = 0; j < old_count; ++j) {
      const size_t base = 1 + kBuildAheadRingFieldsPerReq * static_cast<size_t>(j);
      if (base + kBuildAheadRingFieldsPerReq > mtp_serial_host_ring_.size()) {
        break;
      }
      const int64_t old_id = mtp_serial_host_ring_[base];
      if (cur_ids.count(old_id) == 0) {
        kept.insert(kept.end(), mtp_serial_host_ring_.begin() + static_cast<ptrdiff_t>(base),
                    mtp_serial_host_ring_.begin() + static_cast<ptrdiff_t>(base + kBuildAheadRingFieldsPerReq));
      } else {
        // 本批会覆写该 req：清掉旧 accepted，下面用新值重写。
        mtp_serial_accepted_tokens_.erase(old_id);
      }
    }
  }

  const size_t kept_n = kept.size() / kBuildAheadRingFieldsPerReq;
  const size_t total_n = kept_n + ring_reqs.size();
  InitBuildAheadHostRing(mtp_serial_host_ring_, total_n);
  // 先写保留条目，再写本批新条目。
  for (size_t i = 0; i < kept_n; ++i) {
    const size_t src = i * kBuildAheadRingFieldsPerReq;
    WriteBuildAheadHostRingEntry(mtp_serial_host_ring_, i, kept[src], kept[src + 1], kept[src + 2], kept[src + 3]);
  }
  for (size_t i = 0; i < ring_reqs.size(); ++i) {
    const auto& req = ring_reqs[i];
    WriteBuildAheadHostRingEntry(mtp_serial_host_ring_, kept_n + i, req->req_id, req->generated_tokens[0],
                                 static_cast<int64_t>(req->accepted_tokens.size()),
                                 static_cast<int64_t>(req->forwarding_tokens_draft_num));
    mtp_serial_accepted_tokens_[req->req_id] = req->accepted_tokens;
  }
}

void ModelRunner::WriteBuildAheadRingAfterSampling(const std::shared_ptr<NewModelOutput>& model_output) {
  if (!postprocess_enqueue_report_ || IsMtpSerialFastPath()) {
    return;
  }
  // 非 MTP build-ahead 继续沿用原 device ring: [count, (req_id, token) * B_cur]。
  // MTP 串行路径已在 CPU 侧 repair，不能扩展这里的 ring 协议，否则会破坏 qwen 等模型的原有 kernel 替换。
  // 与 Sampling 同处 PrimaryComputeStream, FIFO 保证先采样后写 ring; B_cur=0 时 kernel 内 no-op.
  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    KLLM_CHECK_WITH_INFO(model_output->cur_pairs_dev != nullptr,
                         "build-ahead enabled but model_output->cur_pairs_dev is null in HandlePostprocess.");
    auto model_sampler = executor_runtime_->GetModelSampler();
    KLLM_CHECK_WITH_INFO(model_sampler != nullptr,
                         "build_ahead_enabled but ModelSampler is null in HandlePostprocess.");
    auto sampler = model_sampler->GetSampler();
    KLLM_CHECK_WITH_INFO(sampler != nullptr, "build_ahead_enabled but Sampler is null in HandlePostprocess.");
    // TP 下仅 rank 0 持有有效 sampled_tokens；非 0 rank 不写 garbage，靠 BroadcastRingSlot 同步。
    if (context_->device->GetDeviceId() == 0) {
      sampler->WriteRing(model_output->slot_index, model_output->cur_pairs_dev, model_output->B_cur,
                         context_->device->GetPrimaryComputeStream());
    }
  }
}

bool ModelRunner::IsMtpSerialFastPath() const {
  return FastPathController::GetInstance().IsEnabledAtStartup() && mtp_step_num_ > 0;
}

void ModelRunner::ClearMtpSerialStateForLifecycle(const std::shared_ptr<ScheduleOutput>& schedule_output) {
  // finish / SyncRecompute：丢弃 Executor 本地 MTP 串行状态（last_block / accepted / frozen drafts / ring 条目）。
  // Engine 已 ResetPrefillingTokens 后若仍用旧 main_pos 裁剪，会把新 prefill 序列裁坏。
  if (schedule_output == nullptr) {
    return;
  }
  auto erase_req = [this](int64_t req_id) {
    mtp_serial_last_block_.erase(req_id);
    mtp_serial_accepted_tokens_.erase(req_id);
    mtp_serial_next_mtp_drafts_.erase(req_id);
  };
  std::unordered_set<int64_t> drop_ids;
  for (const auto& per_dp_vec : schedule_output->finish_req_ids) {
    for (const int64_t req_id : per_dp_vec) {
      erase_req(req_id);
      drop_ids.insert(req_id);
    }
  }
  for (const auto& per_dp_vec : schedule_output->recompute_req_ids) {
    for (const int64_t req_id : per_dp_vec) {
      erase_req(req_id);
      drop_ids.insert(req_id);
    }
  }
  // finish / SyncRecompute 后不再需要保留 host ring 条目。
  if (!drop_ids.empty() && !mtp_serial_host_ring_.empty()) {
    std::vector<int64_t> kept;
    const int64_t old_count = mtp_serial_host_ring_[0];
    for (int64_t j = 0; j < old_count; ++j) {
      const size_t base = 1 + kBuildAheadRingFieldsPerReq * static_cast<size_t>(j);
      if (base + kBuildAheadRingFieldsPerReq > mtp_serial_host_ring_.size()) {
        break;
      }
      if (drop_ids.count(mtp_serial_host_ring_[base]) == 0) {
        kept.insert(kept.end(), mtp_serial_host_ring_.begin() + static_cast<ptrdiff_t>(base),
                    mtp_serial_host_ring_.begin() + static_cast<ptrdiff_t>(base + kBuildAheadRingFieldsPerReq));
      }
    }
    const size_t kept_n = kept.size() / kBuildAheadRingFieldsPerReq;
    if (kept_n == 0) {
      mtp_serial_host_ring_.clear();
    } else {
      InitBuildAheadHostRing(mtp_serial_host_ring_, kept_n);
      for (size_t i = 0; i < kept_n; ++i) {
        const size_t src = i * kBuildAheadRingFieldsPerReq;
        WriteBuildAheadHostRingEntry(mtp_serial_host_ring_, i, kept[src], kept[src + 1], kept[src + 2], kept[src + 3]);
      }
    }
  }
}

void ModelRunner::ApplyCpuBuildAheadRepairFromRing(const std::shared_ptr<ScheduleOutput>& schedule_output,
                                                   const std::vector<int64_t>& prev_ring) {
  // BuildModelInput 前：用上一步 host ring 修补本步占位块。
  // 流程：可选恢复 snapshot → 解析 ph_off → LookupRing → RepairMtpSerialBlock → 更新 last_block / 清 frozen drafts。
  if (schedule_output == nullptr) {
    return;
  }
  if (prev_ring.empty() || prev_ring[0] == 0) {
    return;
  }
  for (auto& req : schedule_output->executor_running_reqs) {
    const InferRequestSerializationSnapshot* snap = FindReqSerializationSnapshot(schedule_output, req->req_id);
    // 分布式 snapshot 恢复（standalone 下 snap 为空，forwarding 由 UpdateInferRequest 的 patch 提供）。
    if (snap != nullptr && snap->placeholder_offset_in_snapshot != std::numeric_limits<size_t>::max()) {
      req->forwarding_tokens = snap->forwarding_tokens;
      req->kv_cached_token_num = snap->kv_cached_token_num;
      req->sampling_token_num = snap->sampling_token_num;
      req->forwarding_tokens_draft_num = snap->forwarding_tokens_draft_num;
    }
    const size_t ph_off = ResolveLaunchPlaceholderOffset(req, snap);
    // 混合 batch（prefill + decode 同批、或新入队 req）可能无占位块；ring 也可能无上一 step 条目。
    if (ph_off == std::numeric_limits<size_t>::max()) {
      continue;
    }
    if (!LookupBuildAheadHostRingEntry(prev_ring, req->req_id, nullptr, nullptr, nullptr)) {
      continue;
    }
    // 占位块 draft 槽数 M = launch 时预留的 forwarding_tokens_draft_num。
    const size_t draft_slots = req->forwarding_tokens_draft_num;
    // 上一块 {main_pos, draft_slots} 取自本地状态而非 req 字段（req 字段已被本步 IPC patch 覆盖，
    // 见 mtp_serial_last_block_ 声明处注释）；首次修补（无记录）视为紧接 prefill，old_draft_slots=0。
    const auto last_block_it = mtp_serial_last_block_.find(req->req_id);
    size_t old_main_pos =
        (last_block_it != mtp_serial_last_block_.end()) ? last_block_it->second.first : req->kv_cached_token_num;
    size_t old_draft_slots = (last_block_it != mtp_serial_last_block_.end()) ? last_block_it->second.second : 0;
    if (old_draft_slots > 0) {
      const size_t expected_ph = old_main_pos + 1 + old_draft_slots;
      if (ph_off != expected_ph) {
        // IPC merge 后 launch placeholder 仍可能与 last_block 不一致：以 ph_off 为准收紧上一块 draft 数。
        if (ph_off > old_main_pos + 1) {
          old_draft_slots = std::min(old_draft_slots, ph_off - old_main_pos - 1);
        } else {
          old_draft_slots = 0;
          old_main_pos = std::min(old_main_pos, req->kv_cached_token_num);
        }
      }
    }
    const std::vector<int>* frozen_mtp_drafts = nullptr;
    const auto frozen_it = mtp_serial_next_mtp_drafts_.find(req->req_id);
    if (frozen_it != mtp_serial_next_mtp_drafts_.end()) {
      frozen_mtp_drafts = &frozen_it->second;
    }
    if (RepairMtpSerialBlock(req, prev_ring, &mtp_serial_accepted_tokens_, ph_off, draft_slots, old_main_pos,
                             old_draft_slots, frozen_mtp_drafts)) {
      mtp_serial_last_block_[req->req_id] = {req->kv_cached_token_num, req->forwarding_tokens_draft_num};
      mtp_serial_next_mtp_drafts_.erase(req->req_id);
    }
  }
}

Status ModelRunner::HandlePostprocessInline(std::shared_ptr<NewModelOutput> model_output) {
  // build-ahead：在 Execute 线程同步跑一轮 HandlePostprocess，与 Forward/WriteRing 同线程串行。
  // 与慢路径 postprocess 线程互斥（Start 时二选一），共享同一套 postprocess 逻辑，无需加锁。
  postprocess_one_shot_output_ = std::move(model_output);
  postprocess_one_shot_ = true;
  postprocess_enqueue_report_ = true;
  return HandlePostprocess();
}

Status ModelRunner::HandleResultReportProcess() {
  SetDevice(context_->device->GetDeviceId());

  while (!terminated_.load(std::memory_order_relaxed)) {
    std::shared_ptr<SamplingOutput> sampling_output = sampling_output_queue_->Get();
    if (!sampling_output) {
      KLLM_LOG_INFO << "HandleResultReportProcess stopped.";
      break;
    }

    // Only device 0 need report sampling output.
    if (context_->device->GetDeviceId() != 0) {
      continue;
    }

    if (context_->global->IsStandalone()) {
      ipc_func_wrapper_->ReportSamplingOutput(sampling_output);
    } else if (context_->global->IsDistributedResultReporter()) {
      sampling_output_pool_->PutSendSamplingOutput(sampling_output.get());
    }
  }
  return Status();
}

}  // namespace ksana_llm
