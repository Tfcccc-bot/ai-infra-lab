/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <vector>

#include "device/device_context.h"
#include "executor/executor_runtime.h"
#include "ipc/ipc_func_wrapper.h"
#include "runtime/model_input.h"
#include "runtime/model_instance.h"
#include "runtime/model_output.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "runtime/weight_instance_inferface.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

class GlobalCachePutWorker;

// Used to execute really inference calculation on the device.
class ModelRunner {
 public:
  explicit ModelRunner(std::shared_ptr<Context> context);

  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);
  void SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetModelWeight(std::shared_ptr<WeightInstanceInterface> model_weight);
  void SetBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> block_allocator_manager);
  void SetSwaBlockAllocatorManager(std::shared_ptr<BlockAllocatorManager> swa_block_allocator_manager);
  void SetHiddenUnitBufferPool(std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool);
  void SetSamplingOutputPool(std::shared_ptr<SamplingOutputPool> sampling_output_pool);
  void SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> draft_generator_controller);
  void SetGlobalCachePutWorker(std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker);

  // Start runner, all worker threads should be ready.
  Status Start();

  // Stop runner, all thread must exit gracefully.
  Status Stop();

 private:
  Status HandlePreprocess();
  Status HandleExecute();
  // 慢路径（build-ahead 关闭）: Execute 只跑 Forward，Postprocess 独立线程做 Sampling。
  Status HandlePostprocess();
  // build-ahead 路径: Execute 内单次调用 HandlePostprocess（不经 model_output_queue_）。
  Status HandlePostprocessInline(std::shared_ptr<NewModelOutput> model_output);
  Status HandleResultReportProcess();
  Status HandleDistributedScheduleOutput();

  void GetHiddenShape(const std::shared_ptr<ScheduleOutput>& schedule_output, std::vector<size_t>& shape);
  DataType GetHiddenDataType();

  // build-ahead + TP: rank 0 WriteRing 后，所有 TP rank 集体广播 ring slot。
  void BroadcastBuildAheadRingSlot(size_t slot);

  // build-ahead: Sampling 后将本步 token 写入 ring slot（仅 postprocess_enqueue_report_ 为 true 时执行）。
  void WriteBuildAheadRingAfterSampling(const std::shared_ptr<NewModelOutput>& model_output);

  // FastPath + MTP 串行：GenerationState 后写 host ring（慢/快路径对比见 .cpp 定义前流程图）。
  void WriteMtpSerialHostRingAfterGenerationState(const std::shared_ptr<NewModelOutput>& model_output);

  // FastPath + MTP>0：Execute 串行 (Sample/Verify/MTP 与慢路径同序); Preprocess 仅入队 schedule,
  // 不做 BuildModelInput. Engine 仍 depth=2 提前 launch, placeholder 由 Execute 读 host ring 修补.
  bool IsMtpSerialFastPath() const;

  // 读上一步 host ring，在 CPU 侧修补 placeholder 与 rejected draft，再同步 BuildModelInput。
  void ApplyCpuBuildAheadRepairFromRing(const std::shared_ptr<ScheduleOutput>& schedule_output,
                                        const std::vector<int64_t>& prev_ring);

  // finish / recompute 时丢弃 Executor 本地 MTP 串行修补状态，避免 Engine 已 Reset 后仍用旧 main_pos 裁剪。
  void ClearMtpSerialStateForLifecycle(const std::shared_ptr<ScheduleOutput>& schedule_output);

  // 在 schedule_output->req_serialization_snapshots 中按 req_id 查找 launch 时刻快照。
  // 分布式路径下用于恢复未被 live req 污染的 forwarding_tokens / kv 等字段。
  static const InferRequestSerializationSnapshot* FindReqSerializationSnapshot(
      const std::shared_ptr<ScheduleOutput>& schedule_output, int64_t req_id);

  // 解析本步占位块在 forwarding_tokens 中的起始下标。
  // 优先级：snapshot.placeholder_offset → req.launch_placeholder_offset → 扫描 kFastPathPlaceholderTokenId。
  static size_t ResolveLaunchPlaceholderOffset(const std::shared_ptr<ExecutorInferRequest>& req,
                                               const InferRequestSerializationSnapshot* snap);

  // build-ahead MTP 串行：Build 前用上一步 host ring 修补占位块 [main | draft×M]。
  // 插入上一 verify 步 accepted draft、回填 main token 与本地 mtp drafts，裁剪上一块残留的被拒 draft
  // （old_main_pos/old_draft_slots 由调用方传入，见 mtp_serial_last_block_ 注释），并同步 kv / sampling 计数。
  // 返回是否实际完成了修补（ring 未命中 / 占位越界等 no-op 场景返回 false，调用方据此决定是否更新状态）。
  static bool RepairMtpSerialBlock(std::shared_ptr<ExecutorInferRequest>& req, const std::vector<int64_t>& prev_ring,
                                   const std::unordered_map<int64_t, std::vector<int>>* accepted_map, size_t ph_off,
                                   size_t draft_slots, size_t old_main_pos, size_t old_draft_slots,
                                   const std::vector<int>* frozen_mtp_drafts = nullptr);

 private:
  std::shared_ptr<Context> context_ = nullptr;

  // The ipc func wrapper, used to invoke functions between processes.
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_ = nullptr;

  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> model_weight_ = nullptr;
  std::shared_ptr<BlockAllocatorManager> block_allocator_manager_ = nullptr;

  // The really executor runtime.
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;

  // Raw pointer to the InflightResourceManager owned by executor_runtime_.
  // Cached here to avoid a shared_ptr copy (and its atomic refcount operations)
  // on the hot preprocess path. Lifetime: executor_runtime_ outlives all
  // worker threads, so this pointer is valid throughout HandlePreprocess.
  InflightResourceManager* inflight_resource_mgr_ = nullptr;

  std::shared_ptr<HiddenUnitBufferPool> hidden_unit_buffer_pool_ = nullptr;

  const size_t mtp_step_num_;

 private:
  // Read schedule_output and build model input.
  std::thread preprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue_ = nullptr;

  std::thread execute_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelInput>>> model_input_queue_ = nullptr;

  // 慢路径: Execute → model_output_queue_ → HandlePostprocess。
  std::thread postprocess_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<NewModelOutput>>> model_output_queue_ = nullptr;

  // build-ahead: HandlePostprocessInline → sampling_output_queue_ → HandleResultReportProcess。
  std::thread result_report_thread_;
  std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue_ = nullptr;

  // build-ahead 单次 postprocess 的跨调用状态（仅 Execute 线程经 HandlePostprocessInline 写入）。
  // 慢路径 postprocess 线程从不触碰这些字段，与 model_output_queue_ 消费路径互斥。
  bool postprocess_one_shot_ = false;
  // true：本步结果入 sampling_output_queue_，由 HandleResultReportProcess 异步上报（build-ahead）；
  // false：postprocess 线程在 device 0 上同步 IPC 上报（慢路径三线程模型）。
  bool postprocess_enqueue_report_ = false;
  std::shared_ptr<NewModelOutput> postprocess_one_shot_output_ = nullptr;

  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  std::atomic<bool> terminated_ = false;

  // 全局 cache put 工作器（Forward 完成后异步上传 KV cache）
  std::shared_ptr<GlobalCachePutWorker> global_cache_put_worker_ = nullptr;

  // 本步采样 ring 的当前 slot；仅在 HandlePreprocess 线程内访问，无需原子。
  // build-ahead 启用时每步 cur_ring_slot_ ^= 1u（0/1 交替），上一步写入 slot 为 cur_slot ^ 1u；
  // 不依赖 batch 是否同构或本步是否做替换（req_id 解耦，prefill/decode 混合一视同仁）。
  size_t cur_ring_slot_ = 0;

  // FastPath + MTP 串行专用 host ring。该路径 BuildModelInput 在 Execute 中串行执行，
  // 因此无需 device ring / CUDA repair kernel；只需保存上一主步的 generated token 供下一步修补。
  std::vector<int64_t> mtp_serial_host_ring_;

  // host ring 额外保存上一 verify 步被 accept 的 draft token 值。标准 ring 只保存 accepted count，
  // 但 MTP 串行 CPU 修补需要把 accepted draft 插回 placeholder 前，才能与慢路径序列长度/kv 起点一致。
  std::unordered_map<int64_t, std::vector<int>> mtp_serial_accepted_tokens_;

  // 记录每个 req 上一次 RepairMtpSerialBlock 修补出的 verify 块 {main_pos, draft_slots}。
  // req->kv_cached_token_num / forwarding_tokens_draft_num 会在每步 UpdateCachedRequests 时被
  // Engine 的 IPC patch 覆盖（Engine 侧数值滞后于 Executor 本地已修补的真实块位置），不能作为
  // "上一块"依据；必须用这个 Executor 本地状态才能正确裁剪被拒 draft、避免 KV 位置漂移。
  std::unordered_map<int64_t, std::pair<size_t, size_t>> mtp_serial_last_block_;

  // 上一步 postprocess 在 MtpForward 结束后冻结的 MTP draft，供本步 repair 回填 draft 槽。
  // repair 不得读 live req->draft_tokens：ring 在 MtpForward 之前写入，而 live draft 在其之后更新，
  // 且下一步 Execute 前 Engine IPC 可能已改写 forwarding，二者与 last_block 几何不同步。
  std::unordered_map<int64_t, std::vector<int>> mtp_serial_next_mtp_drafts_;
};

}  // namespace ksana_llm
