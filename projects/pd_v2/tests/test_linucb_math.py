"""模块 A 单测：LinUCB 数学原语（对齐 KsanaLLM linucb_math.h）。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit.linucb_math import (
    DIM,
    scaled_identity,
    dot,
    matvec,
    quad_form,
    sherman_morrison_update,
    ucb_score,
)


def test_scaled_identity():
    m = scaled_identity(1.0)
    assert len(m) == 16
    for i in range(4):
        for j in range(4):
            assert m[i * 4 + j] == (1.0 if i == j else 0.0)


def test_dot():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert dot(a, b) == 1 * 4 + 2 * 3 + 3 * 2 + 4 * 1


def test_matvec_identity():
    x = [1.0, 2.0, 3.0, 4.0]
    assert matvec(scaled_identity(1.0), x) == x


def test_quad_form_identity():
    x = [1.0, 2.0, 3.0, 4.0]
    assert quad_form(scaled_identity(1.0), x) == dot(x, x)


def test_sherman_morrison_inverse():
    # 从 A0=I 出发，更新得 Ainv = (I + x x^T)^{-1}；应满足 (I + x x^T) @ Ainv ≈ I
    x = [0.5, 0.25, -0.3, 0.1]
    ainv = scaled_identity(1.0)
    sherman_morrison_update(ainv, x)
    A = scaled_identity(1.0)
    for i in range(4):
        for j in range(4):
            A[i * 4 + j] += x[i] * x[j]
    for i in range(4):
        for j in range(4):
            s = sum(A[i * 4 + k] * ainv[k * 4 + j] for k in range(4))
            expected = 1.0 if i == j else 0.0
            assert abs(s - expected) < 1e-9


def test_ucb_initial_bias_only():
    # Ainv=I, b=0 => mean=0, var=x^T x, score=alpha*||x||
    x = [1.0, 0.0, 0.0, 0.0]
    score = ucb_score(scaled_identity(1.0), [0.0] * 4, x, alpha=1.0)
    assert abs(score - 1.0) < 1e-9


def test_ucb_initial_all_ones():
    x = [1.0, 1.0, 1.0, 1.0]
    score = ucb_score(scaled_identity(1.0), [0.0] * 4, x, alpha=2.0)
    assert abs(score - 2.0 * math.sqrt(4.0)) < 1e-9


def test_ucb_with_learned_theta():
    # Ainv=I => theta=b；mean=theta·x，var=x·x
    ainv = scaled_identity(1.0)
    b = [0.2, -0.1, 0.5, 0.0]
    x = [1.0, 0.0, 1.0, 0.0]
    # mean = 0.2*1 + 0.5*1 = 0.7 ; var = 2
    score = ucb_score(ainv, b, x, alpha=0.5)
    assert abs(score - (0.7 + 0.5 * math.sqrt(2.0))) < 1e-9
