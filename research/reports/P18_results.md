# P18 受切换约束的时间活动集合结果

状态：`NEGATIVE`。场景 `cut_roasted_beef`，保留全部 248,895 个实体，以每 kind 原比例的 `0.5 × N × 300` Gaussian-frame cap 约束活动集合，0FT。开发质量为 `{cam01, cam02}` × `T24` 共 48 帧；时间指标为两相机 × 五个 10 帧窗口，共 90 个相邻帧对。方法确定性，已验证种子 0/1/2 的活动矩阵完全一致，每行只评测一次。

## 结果

| 规则 | 活动 Gaussian-frame | cap 利用率 | switches（静态+动态） | PSNR | LPIPS-Alex | worst-10% drop | 标准 TDE | teacher-residual TDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 `gamma>0` | 34,137,056 | 91.436% | 107,883 | **27.127337** | **0.075540** | **11.757787 dB** | **0.00660130** | **0.00036990** |
| R1 `gamma=0` | 33,793,084 | 90.515% | 229,281 | 27.093561 | 0.075618 | 11.799833 dB | 0.00661645 | 0.00045220 |

R0 的切换数比 R1 少 52.95%，teacher-residual TDE 低 18.20%，但注册验收使用的对 GT 残差标准 TDE 只降低 **0.2289%**，远低于 10% 门槛。R0 的 PSNR 仅高 0.033776 dB，亦未达到 +0.10 dB 门槛；LPIPS 和尾部质量没有恶化，但不足以弥补时间机制验收失败。

两个规则都出现注册方案预期的 `budget_gap`。静态组利用率约 97.5%，动态组 R0/R1 仅为 73.78%/70.32%；20 次 λ 二分遇到离散活动跳变后按规定保存最大 feasible 解，不能为追满预算而超 cap 或改阈值。R0 平均每个评测帧活动 113,674 点，R1 为 112,473 点。

## 部署成本

| 项目 | 原模型 | R0 | R1 |
|---|---:|---:|---:|
| 端到端 wall latency | 10.703 ms | 14.154 ms | 13.983 ms |
| 活动 raster GPU 部分 | — | 7.003 ms | 6.927 ms |
| gather GPU 部分 | — | 7.109 ms | 7.015 ms |
| 可部署包 | 原双 PLY 128,586,109 B | 131,121,534 B | 131,379,543 B |

活动子集使 raster 更快，但 gather 抵消收益；R0/R1 的端到端运行时间相对原模型分别增加约 32.25%/30.65%，没有实际部署加速。全部点仍需加载，模型 tensor 为 126,354,304 B；RLE 和可重载元数据还使磁盘包变大。因此不能把 raster-only 数字称为资源改善。

## 正确性与边界

- 30 个 `N≤3, T≤5` 穷举 DP 例的最大目标误差为 `4.44e-16`；全零序列按 tie 规则全部 inactive，空组通过。
- 24 个最近邻 Voronoi bins 完整覆盖 0..299，端点映射正确；RLE 全量重建与活动矩阵位级一致。
- `gamma=0` 结果与独立逐 bin `e > lambda` threshold oracle 完全一致。
- 全活动 gather 相对原模型渲染 max abs 为 0；concat stable ID、每 kind cap、bundle save/load 均通过。
- 时间 ROI 由相邻 GT RGB 差大于 0.03 后 3×3 膨胀得到，只是运动近似；平均 ROI 占 1.548%，不解释为语义动态真值。

该结论只否定当前 T24 粗 bins、K2 删除代理和硬 gather 实现下的注册方法。R0 确实降低了代理切换数和 teacher-residual 变化，但没有转化为验收所需的 GT-residual TDE 或端到端加速，最简单解释是帧间误差主要由 GT/参考变化主导，而每帧重建 CUDA 子模型的成本过高。

原始产物：`runs/research/P18/cut_roasted_beef/active0.5/aggregate.json`。关键实现提交 `70d5b3e` 已推送至 `git@github.com:CLEFFFF2341/4DGS.git` 的 `codex/astra-experiments-20260923`。
