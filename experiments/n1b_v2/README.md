# N1b 重启：时间支持区间积分

从 `archive/n1b-20261020` 新建 `codex/n1b-iteration`。旧 N1b 代码和结果未改；
D1 仍在 `codex/d1-foundation/17e886b`，干净基线仍在 main/cc380cd。

## 结构变更

原 N1b 是解析控制变量加随机残差，存在梯度重尾和代理反向成本。
本版保留“从原始高斯参数推导时间结构”的思路，但移除控制变量反向，改为确定性分区积分。
从原始 4D 协方差计算时间边缘方差，按原生 marginal<0.05 剔除规则得到支持区间。
在活动区间使用 Gauss 积分，完全不活动区间精确加入背景；实际遮挡始终由原 CUDA 渲染。
节点选择 detach，反传只经过原生 renderer。

它是 **有偏确定性积分**，不再声称无偏；梯度目标是原生分支 AD 的期望，
不是包含移动剔除边界项的完整连续曝光导数，也不是对节点生成过程求导。
全时域活跃时退回全局 Gauss；区间数超过节点预算时也回退，不丢弃支持。

首版等分节点在宽错峰反例(seed19)上梯度退化，第二版 balanced 按区间长度/局部 sigma
分配节点。失败结果保存在 confirmation，修正版保存在 confirmation_balanced。
原 N1b narrow_reference 用像素 alpha 事件切分；新方法只用时间 marginal 支持，不偷用参考积分节点。

## 数据与验证范围

- pilot：原窄脉冲固定场景，确认机制可行，保留所有超时。
- confirmation：首版调试 seed0..19；balanced 的 seed17..19 是修复检查。
- confirmation_balanced 的 seed20..83 是独立确认：64 场景随机宽度、相位、错峰和 RGB VJP。
  每方法分别校准 50ms 预算，计入支持构建、排序、Gauss节点构建、前向/反向和 CUDA 同步。
  参考使用独立事件切分4/8阶；超时不计为公平同预算收益。
- controls：旧 dense/static_layers/perspective/static 反例的同调用回退测试；首调冷启动成本保留。
- train_probe：已知曝光的小原生高斯合成拟合，不涉及真实 checkpoint。每方法3秒，
  同 Adam lr=.01，未调参；32或128等节点数，超时末步回滚。高精度评估在计时外且不用于接受步骤。
  提供两份独立估计相乘的 MSE 对照，避免将随机图像 plug-in MSE 的偏差当作唯一收益来源。

联合空间位置/时间宽度拟合的检查视角收益不一致，不能据训练误差宣称几何恢复有效。
`--temporal-only` 是后续独立归因试验：已知固定几何，仅扰动并恢复时间宽度。
该试验不能替代联合拟合失败结果；它只检验条件性时间参数恢复用途。
8x8 ROI 不报告标准 SSIM/LPIPS，不声称真实 coffee_martini PSNR 提升。
目前实现针对小模型；Python 区间处理不适合直接应用于215万高斯，尚未做大模型性能工程。

## 复现

在仓库根目录，顺序执行，避免并发影响计时：

```powershell
$env:PATH="E:\4D_Gaussian\.conda\nerficg\Library\bin;$env:PATH"
.conda/nerficg/python.exe -m unittest experiments.n1b_v2.test_partition -v
.conda/nerficg/python.exe -m experiments.n1b_v2.pilot --sigma .001 --staggered
.conda/nerficg/python.exe -m experiments.n1b_v2.confirm --start 20 --count 8 --allocation balanced
.conda/nerficg/python.exe -m experiments.n1b_v2.controls dense
.conda/nerficg/python.exe -m experiments.n1b_v2.train_probe --seed 3 --method partition --temporal-only
.conda/nerficg/python.exe -m experiments.n1b_v2.train_probe --seed 3 --method gauss --calls 128 --temporal-only
```

场景和方法可分次续跑，JSON保存每个结果；重复相同参数会覆盖对应结果，复测前先提交旧结果。
