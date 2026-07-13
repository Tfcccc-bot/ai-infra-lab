"""投机解码评测入口（Performance or Illusion? 复现，推进中）。

当前状态：复现 throughput-optimal 评测框架，迁移到 EAGLE / MTP / DSpark
的线上参数决策（γ、draft 模型规模、budget 的接受率-开销权衡）。

TODO: 实现接受率 / 端到端加速比 / 系统吞吐变化的实测管线。
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="投机解码评测")
    parser.add_argument("--method", type=str, default="eagle",
                        choices=["eagle", "mtp", "dspark"])
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=4096)
    args = parser.parse_args()
    print(f"[specdec_illusion] TODO: 实现未就绪 (method={args.method}, "
          f"gamma={args.gamma}, max_len={args.max_len})")


if __name__ == "__main__":
    main()
