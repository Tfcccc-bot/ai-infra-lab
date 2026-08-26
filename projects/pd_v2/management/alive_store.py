"""模块 F1：AliveStore + Keepalive —— 对齐 KsanaLLM discovery.cpp §3.3。

Decode 侧 keepalive：
  - 注册键 `pd_v2/decode_alive/<cluster>/<inference_addr>`
  - 值 = `{"epoch": <monotonic 秒>}`（用远端可见的单调时钟，非墙上时间）
  - TTL = 5s；每 2s 续租；关闭时 DEL。
  - 关闭时若失败，等待，重试；下个进程可能重连同一 key（epoch 推进 -> Prefill 视作重启）。

Prefill 侧发现（见 F2）：KEYS `pd_v2/decode_alive/<cluster>/*` 列出存活 decode 地址，
对每个解析 epoch 判断是否存活。

为离线可测：AliveStore 抽象为接口，提供 FakeAliveStore（内存 + TTL）。真实 Redis
后端可在焊接阶段以同一接口注入。
"""

import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


# 对齐 discovery.cpp：键前缀 + TTL
ALIVE_KEY_PREFIX = "pd_v2/decode_alive/"
ALIVE_TTL_SEC = 5.0
ALIVE_RENEW_SEC = 2.0


def alive_key_prefix(cluster: str) -> str:
    return f"{ALIVE_KEY_PREFIX}{cluster or 'default'}/"


def alive_key(cluster: str, inference_addr: str) -> str:
    return f"{alive_key_prefix(cluster)}{inference_addr}"


def encode_alive_value(epoch: float) -> str:
    # 对齐 discovery.cpp：'{"epoch":%lld}'（这里 epoch 为秒级单调值）
    return f'{{"epoch":{int(epoch)}}}'


_EPOCH_RE = re.compile(r'"epoch"\s*:\s*(-?\d+)')


def parse_epoch_from_json(s: str) -> int:
    """解析存活键值的 epoch（对齐 discovery.cpp 的 ParseEpochFromJson）。

    失败时返回 0（= 非存活/无效），与 C++ 行为一致。
    """
    if not s:
        return 0
    m = _EPOCH_RE.search(s)
    return int(m.group(1)) if m else 0


class AliveStore(ABC):
    """对齐 discovery.cpp 中 redis_alive_ 的使用面：Set/Del/Get/Keys/IsConnected。"""

    @abstractmethod
    def set(self, key: str, value: str, ttl_sec: float) -> None: ...

    @abstractmethod
    def get(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def keys(self, prefix: str) -> List[str]: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


class FakeAliveStore(AliveStore):
    """内存实现，支持 TTL（惰性过期）。离线单测用。"""

    def __init__(self, now_fn=None, connected: bool = True):
        self._data: Dict[str, Tuple[str, Optional[float]]] = {}
        self._now = now_fn or time.monotonic
        self._connected = connected
        self._lock = threading.Lock()

    def set(self, key: str, value: str, ttl_sec: float) -> None:
        with self._lock:
            expire_at = None if ttl_sec is None else self._now() + ttl_sec
            self._data[key] = (value, expire_at)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._expire()
            e = self._data.get(key)
            return e[0] if e else None

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def keys(self, prefix: str) -> List[str]:
        with self._lock:
            self._expire()
            return [k for k in self._data if k.startswith(prefix)]

    def is_connected(self) -> bool:
        return self._connected

    def _expire(self) -> None:
        now = self._now()
        for k, (_, exp) in list(self._data.items()):
            if exp is not None and exp <= now:
                del self._data[k]


class KeepaliveAgent:
    """Decode 侧存活键守护（对齐 discovery.cpp DecodeEngine::RunKeepalive）。

    提供显式 register()/renew()/shutdown() 步骤（便于离线单测），同时支持
    start()/stop() 后台续租线程（焊接到真实运行时使用）。
    """

    def __init__(
        self,
        store: AliveStore,
        cluster: str,
        inference_addr: str,
        epoch_fn=None,
        ttl_sec: float = ALIVE_TTL_SEC,
        renew_sec: float = ALIVE_RENEW_SEC,
    ):
        self._store = store
        self._key = alive_key(cluster, inference_addr)
        self._ttl = ttl_sec
        self._renew = renew_sec
        self._epoch = epoch_fn or time.monotonic
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # 显式步骤（单测直接调用）----

    def register(self) -> None:
        # 先删旧键再写新键（epoch 推进）
        self._store.delete(self._key)
        self._store.set(self._key, encode_alive_value(self._epoch()), self._ttl)

    def renew(self) -> None:
        self._store.set(self._key, encode_alive_value(self._epoch()), self._ttl)

    def shutdown(self) -> None:
        self._store.delete(self._key)

    # 后台守护（焊接期使用）----

    def start(self) -> None:
        if self._thread is not None:
            return
        self.register()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._renew):
            if not self._store.is_connected():
                continue
            try:
                self.renew()
            except Exception:
                # 对齐：关闭失败 -> 等待后重试（这里简单吞掉，重启循环）
                pass

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._renew + 1.0)
        self.shutdown()
        self._thread = None
