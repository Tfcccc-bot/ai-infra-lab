"""LinUCB 线性代数原语（对齐 KsanaLLM csrc/pd_v2/decode/linucb_math.h）。

纯 Python / 标准库实现，维度固定为 4（bias + idle + tpot_headroom + remaining_small）：
- 向量是 length-4 的 list[float]
- 矩阵是 row-major 的 length-16 的 list[float]

所有函数无状态、无外部依赖，便于离线单测与后续移植到 sglang fork。
"""
from __future__ import annotations

DIM = 4


def scaled_identity(value: float) -> list[float]:
    """A0 = lambda*I 的逆 = (1/lambda)*I；这里 value 即 1/lambda（如 lambda=1 => I）。"""
    m = [0.0] * (DIM * DIM)
    for i in range(DIM):
        m[i * DIM + i] = value
    return m


def dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(DIM))


def matvec(a: list[float], x: list[float]) -> list[float]:
    return [sum(a[i * DIM + j] * x[j] for j in range(DIM)) for i in range(DIM)]


def quad_form(a: list[float], x: list[float]) -> float:
    """x^T A x，用于 LinUCB 探索 bonus 的 sqrt(x^T Ainv x)。"""
    return dot(x, matvec(a, x))


def sherman_morrison_update(ainv: list[float], x: list[float]) -> None:
    """对存储的逆做秩 1 更新：Ainv <- (A + x x^T)^{-1}。

    等价 Ainv - (Ainv x)(Ainv x)^T / (1 + x^T Ainv x)，原地修改 ainv。
    denom >= 1（PSD Ainv + 实 x），数值安全。
    """
    u = matvec(ainv, x)  # Ainv * x（对称时等于 x^T Ainv）
    denom = 1.0 + dot(x, u)
    for i in range(DIM):
        for j in range(DIM):
            ainv[i * DIM + j] -= (u[i] * u[j]) / denom


def ucb_score(ainv: list[float], b: list[float], x: list[float], alpha: float) -> float:
    """单臂 UCB 分数：mean + alpha * sqrt(var)。

    mean = theta·x（theta = Ainv b），var = x^T Ainv x（PSD，向下截断到 0）。
    """
    theta = matvec(ainv, b)
    mean = dot(theta, x)
    var = max(0.0, quad_form(ainv, x))
    return mean + alpha * var ** 0.5
