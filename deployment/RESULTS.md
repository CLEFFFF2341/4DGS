# coffee_martini 完整实验结果

> 全流程已完成；尚不能认定论文各项数值指标已复现。论文表 11 是三个场景的汇总，本次仅 coffee_martini。核查见 PAPER_REPRODUCTION_AUDIT.md。

- 输出目录：`E:\4D_Gaussian\output\FasterGS4D\fastergs4d_coffee_martini_2026-10-18-11-20-32`
- 官方 FasterGS4D 分支：`3dbbb985c5e0f29825d0960563f3c39f29eb3987`
- NeRFICG：`e458ac6e41810ad6d9405687c1e698fadd28dac5`
- 训练：30,000 次迭代，batch size 4，SH degree 3，随机种子 0。
- 分辨率：1352×1014；每相机 300 帧、30 fps。
- 划分：17 路相机共 5100 张训练图；cam00 共 300 张测试图。
- 初始稠密点云：589,789 点；原始视频、图像、相机参数均保留。
- 最终高斯数量：2,150,292。
- 无损缓存：5400 张 float32 图像全部通过与官方加载器的精确数值相等检查。

## 完整测试视角评估

- PSNR: 26.321117
- SSIM: 0.893975
- LPIPS: 0.212569
- 指标基于官方 NeRFICG 的 PNG 8-bit 输出计算，包含所有 300 帧。

## 耗时与产物

- 官方计时器训练时间：44.48 分钟（不含预处理、缓存及后处理）。
- float32 缓存准备时间：117.88 秒。
- 训练峰值显存：allocated 3.50 GiB；reserved 6.27 GiB。
- 测试指标：`E:\4D_Gaussian\output\FasterGS4D\fastergs4d_coffee_martini_2026-10-18-11-20-32\inference\test_30000\metrics_8bit.txt`
- 动态视频：`E:\4D_Gaussian\output\FasterGS4D\fastergs4d_coffee_martini_2026-10-18-11-20-32\coffee_martini_cam00_dynamic.mp4`
- 视频已解码核验：1352×1014，300 帧，30 fps，10 秒。
- 最终检查点已核验：30000 次迭代，所有浮点参数有限。

环境版本、Windows 编译适配和 LPIPS 权重来源详见同目录 README.md。
