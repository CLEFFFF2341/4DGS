# P03 时间聚合与短暂事件诊断结果

状态：`COMPLETED / NOT_ADVANCED`。S0/M0、b=0.5、0FT；全部规则确定性。R0 与 P02 hash/选择一致，正式质量结果复用；R1/R2 各评测一次。

| 规则 | 聚合 | PSNR | LPIPS-Alex | worst-10% drop | incremental TDE | teacher-residual TDE |
|---|---|---:|---:|---:|---:|---:|
| R0 | mean | 27.250421 | 0.075145 | 11.675420 dB | 1.12505e-5 | 3.28181e-4 |
| R1 | max | 27.395662 | 0.073688 | 11.399868 dB | 1.21199e-5 | 3.11571e-4 |
| R2 | top-3 mean | 27.384616 | 0.074017 | 11.413717 dB | 9.78917e-6 | 3.09916e-4 |

max 比 mean 高 0.145242 dB、LPIPS 好 0.001457、尾部 drop 好 0.275553 dB，说明强调短时峰值有小幅真实收益；top-3 与 max 质量接近。R2 的 incremental TDE 比 R0 低约 13.0%，但 teacher-residual TDE 只低约 5.6%，没有达到 10% 时间机制门槛。

三规则都仍落后 P01-R4 的 27.802153 dB；最佳 R1 低 0.406491 dB，且尾部更差 0.588031 dB，所以不晋级。K2 标签统计为短暂支持点 12,757、持续点 170,602、零支持点 33,171。

奇偶各 12 个 T24 时间的排序稳定性较高：R0/R1/R2 Spearman 分别为 0.9874/0.9811/0.9850，top-K Jaccard 为 0.9584/0.9468/0.9545。固定窗口 GT 使用两次像素 SHA256 一致才接受的共识缓存；两相机各新增 46 帧，最多 3 次解码尝试。重算后各规则 reference TDE 与 GT frame difference 完全一致，消除了首次 H.264 解码不稳定。

原始结果位于 `runs/research/P03/cut_roasted_beef/`。
