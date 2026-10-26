# N1b 重启：时间支持区间积分

2026-10-25 联合优化复查：新增 `joint_probe.py`、尺度归一化参数更新及完整图像
拟合路径。原实验结果保留；诊断、确认协议和新结果见
[完整复查与修复报告](JOINT_REPAIR.md)、[联合优化修复协议](JOINT_REPAIR_PROTOCOL.md)
与 `results/joint_audit/`。单长曝光的图像收益并未普遍转化为几何恢复；
额外多曝光观测在新的四例合成确认中同时改善瞬时图像和几何，但普通Gauss8更好。
原 `train_probe` 的默认调用仍用于复现历史 ROI 协议；修复后的联合拟合入口：

```powershell
$env:PATH="E:\4D_Gaussian\.conda\nerficg\Library\bin;$env:PATH"
.conda/nerficg/python.exe -m experiments.n1b_v2.joint_probe --seeds 6 --full-frame --arms partition:32:scale-aware:0.01:0.05 --output experiments/n1b_v2/results/my_joint_run.json
```

`--arms` 字段依次为积分方法、节点数、位置参数化、空间学习率、时间学习率。
`--steps` 可改为相同步数诊断；`--views 2` 增加第二个训练相机；
`--initialization xyz` 同时扰动三个位置坐标。输出文件已存在时拒绝覆盖。
新入口仍是已知曝光的合成小模型实验，未接入真实数据 Trainer。

多曝光联合拟合与单长曝光的对照使用：

```powershell
.conda/nerficg/python.exe -m experiments.n1b_v2.sequence_probe --seeds 14 15 --output experiments/n1b_v2/results/my_sequence_run.json
```

通用曝光接口为 `Exposure(lo, hi)` 和 `predict_exposure(..., method='gauss'|'partition')`；
必须提供实际快门范围。多曝光试验使用了额外观测，详见
[多曝光协议](SEQUENCE_PROTOCOL.md)，不能将其收益算作同数据下的积分器优势。

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

## 真实曝光数据验证（CDAVIS）

已下载真实 Ev-DeblurNeRF-CDAVIS，核验逐帧约 100 ms 曝光，完成五场景、三个种子的公平对照。
修改优化设置相对原设置平均提升 2.04 dB，但相对同优化器 Gauss 仅差 0.00048 dB，没有实际采样优势。
这是静态场景相机运动模糊的共同初始化细化实验，不是动态 coffee_martini 的结果。
详见 [结果与限制](REAL_CDAVIS_RESULTS.md)、[固定协议](REAL_CDAVIS_PROTOCOL.md) 和 [原始复核数据](results/real_cdavis/audit.json)。

## 真实动态场景验证（DSEC）

已下载 DSEC 两个真实序列的三个动态片段，使用逐帧曝光起止和双目标定，完成三个种子、27 组方法对照。
修改版相对原设置的全图 PSNR 在两个日景片段为 −0.023/−0.160 dB，夜景为 +0.346 dB；相对公平 Gauss 平均 −0.0009 dB，没有稳定额外收益。
实际规模的支撑区间触发全局回退，全部训练步骤使用与 Gauss 相同的节点。
详见 [动态结果与边界](REAL_DYNAMIC_RESULTS.md)、[固定协议](REAL_DYNAMIC_PROTOCOL.md) 和 [复核数据](results/real_dynamic/audit.json)。

## 原 FasterGS4D 核心配置补测

相同 DSEC 初始化下，单时刻原渲染、FusedAdam 与原 L1+DSSIM 损失在等渲染次数时达到 31.88 dB，修改版为 30.46 dB，9/9 次更高。
完整 Trainer 被 Windows 应用控制阻止加载 MortonEncoding；本轮是固定容量核心适配，未运行完整默认增密流程，也未证明等时长优势。
详见 [结果与执行范围](FASTERGS_CORE_RESULTS.md)、[协议](FASTERGS_CORE_PROTOCOL.md) 和 [原始审计](results/fastergs_core/audit.json)。
