# P20 SH 精度分配依赖阻塞

状态：`BLOCKED_CACHE`，不是负结果。正式运行数为 0，未生成或填入预期质量指标。

P20 的注册失真要求在冻结原模 ray 上计算
`D_io = mean_r ||w_ir * (evalSH_original - evalSH_option)||²`。这至少需要每条采样 ray 的方向、原始点贡献权重 `w_ir`、点 ID/顺序和颜色重放信息。当前缓存状态为：

- K0 可用，包含点属性与 T24 几何；
- K1 可用，但只有按视时聚合的贡献、hit、tile touches 等量；
- K2 可用，但只有平方后的单删误差 `e_it`，有符号 RGB 与 ray 方向已经丢失；
- K3 未实现，`research_cache` 中没有 `k3.pt`、有序 ray list 或等价 CSR 颜色缓存。

因此无法从 K1/K2 精确恢复五个 SH option 的 `D_io`。用聚合贡献乘 SH 系数能量、用 dominant ID、或把 K2.E 当作 SH 敏感度都会改变 Astra 固定的方法定义，也无法通过“SH eval 与 CUDA 三方向一致”的正确性检查。本轮没有采用这些替代。

解锁 P20 需要先完成共享 K3：CUDA 导出完整有序 contributor/ray direction/color 信息，按 8×8 patch 的 16 个固定像素建立 CSR/chunk 缓存，并通过原 renderer 重放、采样误差和容量门槛。该基础设施也会同时解锁 P04/P08/P09；它不属于 P20 单行的一次普通实现修复。

阻塞产物：`runs/research/P20/cut_roasted_beef/shac0.5/status.json` 与 `dependency_audit.json`。
