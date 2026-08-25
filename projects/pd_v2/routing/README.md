# routing/

P/D pair 策略与调度模拟（M3）。

SGLang 原生 Router 负责选 Prefill/Decode 并生成 `bootstrap_room`；本项目在其上叠加**拓扑感知的 pair 评分**。

## 计划落地

- `pair_score.py` —— 把拓扑距离纳入 P/D 选择（结合 topology/score.py）
- `simulate.py` —— 用离散事件模拟验证策略，先于真实集群跑，快速看出“auto 是否选了明显更差路径”
- 后续 M4 把 transfer queue 水位、decode 可用 KV、extra slots 纳入 admission 控制

## 原则

- 先模拟、后上真机；评分函数保持可插拔（策略与机制分离）。
- 评分输出要能打 metric，便于 M6 的 A/B 对比。
