"""Kitty 2-bit KV Cache 量化复现入口（推进中）。

当前状态：复现 Kitty 的动态通道精度提升（dynamic channel-wise precision boost），
并与 Momenta BBQ 概率积分变换量化结合，用于 KsanaLLM 长上下文 KV Cache 压缩。

TODO: 实现通道重要性评估、KV INT2 重排与 INT8 baseline 对比评测。
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Kitty 2-bit KV Cache 复现")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kv_bits", type=int, default=2)
    args = parser.parse_args()
    print(f"[kitty_kv2bit] TODO: 实现未就绪 (model={args.model}, kv_bits={args.kv_bits})")


if __name__ == "__main__":
    main()
