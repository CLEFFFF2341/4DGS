# P99 阶段一汇总状态

状态：`COMPLETED`。P01–P20 共 20 个方案 ID 全部进入主表：

- `ADVANCED_STAGE1`：2 个（P15、P19）；
- `BLOCKED_CACHE/BLOCKED_TARGET`：5 个（P04、P08、P09、P14、P20）；
- `PROXY_INVALID`：1 个（P06）；
- 正确实现后的 `NEGATIVE`：8 个（P07、P10、P11、P12、P13、P16、P17、P18）；
- 完成但不晋级：3 个（P02、P03、P05）；
- 必要基线：1 个（P01）。

阶段一累计账本为 0.8705 GPU 小时、0.5262 CPU 选择小时，共使用 4/4 个共享资源控制槽。cam00 最终测试仍封存。

## 阶段二候选

1. P15-R0，自适应 position+rotation knots；
2. P19-R1，按 tile cost 的点选择；
3. P16-R1，作为 FP16 position 简单控制候选。

P01-R4 与 P02-R0 是阶段二必带控制。P19-R0 虽胜内部同 byte 纯 E 控制，但在几乎相同 bundle bytes 下比 P01-R4 低 0.416065 dB，故 P99 公平性复核没有选择它。

阶段二未启动：`research/progress.json` 只授权阶段 0/1，且 S1 `coffee_martini`、S2 `sear_steak` 缺少可用的官方预训练 checkpoint 与完整预处理。不能用 S0 重复种子替代跨场景验证。

完整产物位于 `runs/research/aggregate/20260927-stage1-a5e2cda/`：`all_runs.csv`、`per_scene_budget.csv`、`pareto.csv`、`fairness_audit.csv`、`budget_ledger.csv`、`failure_catalog.md`、`report.md`、`next_queue.json`、`fixed_visuals/` 与 `astra_handoff.md`。汇总代码为提交 `b30771f`。
