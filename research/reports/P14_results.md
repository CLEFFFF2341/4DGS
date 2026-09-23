# P14 动态轨迹向线性静态表示回收结果

状态：`BLOCKED_TARGET`。日期：2026-09-24。严格使用预注册 opacity 与 2° rotation gate，未降低门槛。

- 原动态点：63,862。
- opacity gate 通过：7,186。
- rotation gate 通过：13。
- 两门交集：13（0.02036%）。
- 每转换点实际 payload 差：约 1,000 bytes。
- 目标节省：31,931,000 bytes；最大可达：13,000 bytes，仅为目标约 0.0407%。

按协议将全部 13 个合格点转换为 `xyz=a, xyz_disp=b, rotation=q149, opacity=base` 并评测可达 RD 点。转换后 N 保持 248,895，V×T24 的 PSNR 为 35.651642 dB，teacher MSE `3.55e-11`，平均 PSNR drop `5.56e-7` dB；说明实现数值影响可忽略，但可转换数量远不足以形成目标码率点。

R0 几何误差排序与 R1 运动幅度排序在只有 13 个合格候选且全部都需转换时得到同一集合，因此不生成两份伪独立 run。结论是当前 Ex4DGS 模型的动态旋转已充分区分，P14 在冻结保护条件下不可达目标，而不是方法质量负结果。完整产物位于 `runs/research/P14/`。
