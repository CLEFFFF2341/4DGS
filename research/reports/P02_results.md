# P02 单删除反事实误差基线结果

状态：`COMPLETED / NOT_ADVANCED`。S0/M0、b=0.5、0FT，确定性 seed 0/1/2 ID 不变性通过后正式评测一次。

- K2 key：`d5073d9d688b15063013e59bf67524819aa99f033b67c6e3f2d0967542788014`。
- 保留 124,447 点（static 92,516 / dynamic 31,931），215,724 个点的 mean 单删分数非零。
- PSNR 27.250421 dB，SSIM 0.934411，LPIPS-Alex 0.075145。
- mean PSNR drop 8.401222 dB，worst-10% drop 11.675420 dB。

P02 比 P01 最佳简单行 R4 的 27.802153 dB 低 0.551732 dB，LPIPS 差 0.000190，worst-10% drop 差 0.863583 dB。K2 与真实单删已经数值一致，因此这里不是代理实现错误；它表明独立单点删除 MSE 在同时删除约一半点时缺少联合交互建模，不能保证最好的多删集合。P02 作为精确单删强基线保留，但不晋级。

原始结果位于 `runs/research/P02/cut_roasted_beef/`。
