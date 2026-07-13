"""P/D 分离 break-even 分析入口（Beyond the Buzz 祛魅，推进中）。

当前状态：复现 P/D 分离部署决策框架，结合 KsanaLLM 真实业务负载
（chat / 长文档 / code）做 P/D 分离 vs 混合部署的 break-even 分析。

TODO: 实现不同负载下的 TTFT / TPOT / KV 传输开销占比评测与拐点分析。
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="P/D 分离 break-even 分析")
    parser.add_argument("--workload", type=str, default="chat",
                        choices=["chat", "longdoc", "code"])
    parser.add_argument("--kv_bw", type=float, default=50.0,
                        help="KV Cache 传输带宽 (GB/s)")
    args = parser.parse_args()
    print(f"[pd_disaggregation] TODO: 实现未就绪 (workload={args.workload}, "
          f"kv_bw={args.kv_bw})")


if __name__ == "__main__":
    main()
