# Faster-GS 原论文复现核查

核查日期：2026-10-18。结论：**coffee_martini 的训练、评估和视频产物完整，但尚不能认定论文各项数值指标已经复现。** `results.json` 中 `status: complete` 仅表示本次工作流完成；`paper_reproduction_status` 单独记录论文复现状态。

## 原文对应关系

核查原文：[Faster-GS 论文](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf)，附录 C.5、表 11（PDF 第 18 页）。该表覆盖 Coffee Martini、Cook Spinach、Flame Steak 三场景，Ours 行为 PSNR 30.54、训练 18m57s、显存 3.7 GiB，未列真实动态场景的 SSIM/LPIPS。正文表 5 的 SSIM/LPIPS 属于合成 D-NeRF，不能用作 coffee_martini 的目标。论文真实场景设置为 1352×1014、提供的训练/测试划分、SH 三阶、30,000 次迭代、batch 4，以及预处理初始点云。

## 逐项判定

| 项目 | 本次实际值 | 核查判定 |
|---|---|---|
| PSNR | 26.321117 dB，cam00 全部 300 帧的逐帧均值 | 计算口径经独立复算验证；不能与三场景汇总值直接判定达标 |
| SSIM | 0.893975 | 官方评估器真实输出；表 11 没有对应参考值，无法判定论文数值复现 |
| LPIPS-VGG | 0.212569 | 官方评估器使用 VGG、normalize=True；真实场景论文未给参考值，无法判定数值复现 |
| 训练时间 | 2669.079451 秒，即 44m29.079s | 为 coffee 单场景计时，不是同一三场景统计；不能宣称复现论文速度 |
| 峰值 allocated 显存 | 3.497159 GiB | torch.cuda.max_memory_allocated；不是三场景统计，不能因接近论文值就宣称复现 |
| 峰值 reserved 显存 | 6.267578 GiB | 缓存分配器保留值，与 allocated 分开报告；不是 nvidia-smi 整卡占用 |
| 最终高斯数 | 2,150,292 | 本次 checkpoint 真实记录；表 11 没有对应参考值 |
| 动态视频 | 1352×1014，300 帧，30 fps，10 秒 | 产物规格通过 ffprobe 解码核验；视频存在不等于论文图像质量达标 |

本次 PSNR 与表 11 Ours 数字的算术差约为 -4.22 dB，**不能解释为 coffee_martini 相对作者同场景下降 4.22 dB**，因为原文没有在该表拆出 coffee 单场景值。表中汇总的具体逐场景/多次运行明细也未给出，不应自行补造。

## 已核实的实验设置与输出

- 配置：`configs/fastergs4d_coffee_martini.yaml`；实际 checkpoint 为 30,000 次迭代，参数全为有限值；batch 4、SH 三阶。
- 原始视频 2704×2028，每路 300 帧、30 fps；官方加载器以 0.5 缩放到 1352×1014，没有减少帧数。
- 17 路训练相机共 5100 张；测试 cam00 共 300 张；训练/测试文件集合不相交。初始稠密点云使用训练相机的首帧，不使用测试相机。
- 固定 cam00 测试视角，按原始时间顺序渲染全部 300 帧；报告没有改用训练视角，也没有只选择好看的帧。
- 当前 FasterGS4D 源码工作树干净，commit `3dbbb985c5e0f29825d0960563f3c39f29eb3987`；框架 commit `e458ac6e41810ad6d9405687c1e698fadd28dac5`。
- 官方 `src/Methods/Base/Renderer.py` 从保存的 8-bit PNG 重读评估，PSNR/SSIM 的 data_range 为 1；LPIPS 使用 VGG 并对 [0,1] 输入进行规范化。报告采用逐帧算术均值，没有把 PixelMean 或训练损失混为测试指标。
- `deployment/audit_saved_metrics.py` 用 NumPy float64 对 300 对 PNG 独立复算 PSNR，并抽查第 0、150、299 帧 GT 与原始 cam00 图像经过官方加载器缩放及 PNG 量化后的逐像素一致性；结果见 `metric_audit.json`。SSIM/LPIPS 本次核查了官方实现及既有完整评估输出，没有另写一个独立实现重算全部帧。
- LPIPS 的 VGG 特征权重来源、固定版本及转换验证见 `lpips_weights.json`、`vgg16_archive.json`；未使用随机 VGG 权重。
- 最终保存模型重新加载后完成了完整测试集评估，结果与训练结束评估的报告精度一致，见 `reload_render_validation.json`。

## 仍不满足严格可比性的因素

1. 目前只运行一个场景，缺少 Cook Spinach、Flame Steak，无法复算表 11 的三场景统计。没有运行 Yang 基线，因此也未验证论文所述相对加速与显存降低。
2. 初始点云是本机 COLMAP 4.2.1 重建的 589,789 点。论文描述了预处理思路，但在已核查的论文、项目主页和当前官方分支中未找到作者对应的 coffee 单场景指标、固定点云及完整实验配置。示例配置不能替代作者实验配置的逐项证明。
3. 本次 Windows/WDDM、Anaconda PyTorch 2.8.0 CUDA128、cuDNN 92600；与当前框架说明的测试 cuDNN 91002 不同。不能把当前代码、软件栈等同于论文实验时的软件栈。
4. 本次 PRELOADING_LEVEL=0，采用磁盘上的 float32 无损缓存。5400 张缓存与官方加载器输出逐值相同，故没有发现输入量化变化；但加载路径会影响时间，不能据此声明性能完全可比。所报训练时间排除初始数据加载、预处理、缓存构建及后处理，但包含训练步骤中的数据读取/传输。
5. 论文第 4.1 节的一般实验说明提及预上传训练图像、从计时中排除上传且计入显存；该节主要介绍 3D 实验，不能未经证据直接断言真实 4D 实验也把全部图像放入显存。当前材料不足以把其 4D 峰值显存与本次 allocated/reserved 精确对齐。

以上是可比性限制，**不是已经证实的 PSNR 差异原因**。未做消融实验，不能把差异归因于 Windows、CUDA、缓存或点云中的某一个因素。

## 后续达到论文复现结论所需证据

获得作者三场景逐项参考结果、确切配置与初始点云/预处理版本；完成另外两个场景，统一统计后与表 11 比较。若要验证加速与显存优势，还需在同一机器、输入、加载和计时口径下运行 Yang 基线。真实场景 SSIM/LPIPS 可以继续作为本次实测指标，但在没有作者参考值时不能标注“与论文一致”。

本次核查没有修改模型、渲染结果或实测指标，也没有启动额外训练。
