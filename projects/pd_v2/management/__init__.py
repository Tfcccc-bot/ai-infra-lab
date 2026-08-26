"""pd_v2 整套管理（对齐 KsanaLLM csrc/pd_v2）。

模块划分（与 KsanaLLM 真实实现一一对应）：
  E   - PeerSelector：快照驱动的 P 节点路由表（peer_selector.{h,cpp}）
  F1  - AliveStore + Keepalive：Redis 存活键 + epoch JSON（discovery.cpp §3.3）
  F2  - 快照发现/广播 + 入站 reverse-route 配对（OnPrefillSnapshot/BroadcastSnapshot）
"""
