# P15 自适应轨迹关键帧迁移结果

状态：`ADVANCED_STAGE1`。场景为 `cut_roasted_beef`，开发视角 `{cam01, cam02}` × `T24` 共 48 帧，1352×1014，0FT。R0/R1 都是确定性方法，seed 0/1/2 不变性检查通过后各评测一次。

## 结果

变量部分为动态 position+rotation，原始 66,161,032 B；目标 33,080,516 B。每个 knot 使用 position FP32、规范化 quaternion FP32 和逻辑 uint16 索引，另计每点 offset 与 header。两行均实际使用 33,080,494 B、1,094,167 knots，平均每点 17.1333/37 knots；可加载紧凑包均为 95,270,974 B。

| 规则 | 分配 | PSNR | SSIM | LPIPS-Alex | mean drop | worst-10% drop | position 相对误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| R0 | 全局最大 position+rotation 误差自适应 | 35.119142 | 0.963235 | 0.044877 | 0.532501 dB | 1.465721 dB | 0.007302 |
| R1 | stable-ID 轮转、最长区间均匀二分 | 32.578974 | 0.953974 | 0.052056 | 3.072669 dB | 7.507267 dB | 0.013657 |

R0 在完全相同 payload 下比 R1 高 2.540168 dB，LPIPS 好 0.007179，worst-10% drop 好 6.041545 dB；因此自适应分配的收益不是由更多 bytes、更多点或牺牲尾部换均值造成。R0 的最差帧为 cam01/t273，下降 1.695843 dB。

## 正确性与成本

- 线性位置和常旋转只用端点即可精确恢复；`q` 与 `-q` 的旋转角误差为 0。
- 180° shortest-arc slerp toy 最大误差为 0；all-knots 模式保持原 position/rotation tensor bit-exact。
- knot 索引均在 padding 范围内；本 Torch 版本用同宽 `int16` 容器保存 J=37 的逻辑 `uint16`，不改变数值范围或字节数。
- 通用解码 bundle 与紧凑 bundle 在 t=149 和真实 fractional t=0.5 的重载渲染最大差为 0，stable IDs 不变。
- 128 轨迹 smoke 完成 1,935 次增量选择；全量 R0 完成 966,443 次，选择 45.769 秒，选择+解码 67.072 秒，总运行 110.348 秒。最终被接纳 knot 的联合误差从 0.99987 降到 0.005078。
- 该表示加载后恢复原 dense FP32 tensors，只主张磁盘表示收益，不主张显存或渲染加速。

## 同字节剪枝控制与当前判断

同上限 P01-R4 经 8 次二分字节探测后保留 181,324 点，bundle 为 95,132,216 B，PSNR 28.372682、LPIPS 0.067532、worst-10% drop 10.380492 dB。P15-R0 分别改善 6.746460 dB、0.022654 和 8.914771 dB，因此通过 Astra 阶段 1 的公平控制门槛，更新为 `ADVANCED_STAGE1`。

这一结论仍限于 S0 开发视角的零微调结果；它不等于跨场景、最终 cam00 或统计显著性已经得到证明。P15 的变量预算与只压 position 的 P16/P17 不同，也不能仅按 PSNR 作跨赛道总排名。

原始结果位于 `runs/research/P15/cut_roasted_beef/`。
