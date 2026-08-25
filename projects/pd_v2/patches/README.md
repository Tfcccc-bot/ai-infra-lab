# patches/

指向 SGLang fork 的 commit / 分支，**不复制大段源码**。

## 约定

- 每个上游候选改动是一个小 commit，映射到本目录一个 `.md` 文件：
  - `topology-discovery.md`
  - `affinity.md`
  - `routing-policy.md`
  - `metrics.md`
  - `tests.md`
- 每个文件记录：SGLang fork 分支、commit hash、改动意图、测试结果、是否已提上游 PR。

## 合规提醒

本仓库不得包含腾讯内部文件路径、内部提交哈希、专有类实现或原始代码片段。核心实现只存在于 SGLang fork（或未来的上游 PR），这里只留指针与说明。
