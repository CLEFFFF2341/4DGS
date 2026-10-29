# MVTV 真实数据开发训练（2026-10-30）

用户授权启动真实数据训练后，已下载官方 MVTV，完成 Road 场景审计、真实 CUDA 试跑和两次有限预算训练。最终采用分离 RGB 训练遮罩与保护遮罩的版本。运行已正常完成，当前没有留下持续训练进程。

## 数据与协议

- 来源：[作者仓库](https://github.com/XuJiacong/Thermal-NeRF)提供的 MVTV 压缩包，1,417,015,707 bytes；SHA256 与下载地址见 `results/dataset_source.json`。本地归档 `downloads/rgb_thermal/mvtv/mvtv.zip`，已解压 10 个 VS 场景至 `dataset/rgb_thermal/mvtv/vs/`。
- 使用 Road 的官方 24 train / 2 eval 划分，保留 frame_00013 和 frame_00015。评估图像不参与梯度、步长选择或候选提交。
- 独立 reviewer 审计原 COLMAP 相机和点云，中位重投影误差约 0.562 px，坐标一致。沿用原 COLMAP 世界系、OPENCV 内参与畸变；全部模态采用相同去畸变和缩放。
- 作者 SfM 使用全部 26 个视角，属于 **transductive SfM**，不是严格 train-only pose/初始化。886 个点中 870 个至少有 2 个训练观测，用于本次初始化；初始颜色重新从训练 RAW 轨迹采样，不使用 COLMAP 自带颜色。筛点并没有消除全视角 BA 的信息。
- 热图是作者通过单应变换配准的 8 位 JPEG 灰度。本次使用 **author_homography_proxy / processed intensity** 开发协议，不宣称真实独立双相机辐射标定，未将它伪装成 strict `linear_sensor` manifest。
- RGB 使用分发的 16 位 RAW 派生 PNG，除以 65535 及只由训练图计算的固定 p99。黑电平、精确曝光比未得到验证；noise 数值是固定 robust-loss 超参数，不是实测传感器噪声。
- 静态曝光中点近似；没有真实运动去模糊训练。长曝光图及 brights_gt 不用于训练或评估，本轮只报告短曝光观测重建误差。不能套用相机 UI 的 1/1 和 1/64 推出曝光比64。

## 已完成的运行

最终目录：`output/thermal_supervision/mvtv_road_fullrgb_pilot_20261030/`。

| 项目 | 设置/状态 |
|---|---|
| 图像尺寸 | 284 × 211 |
| 高斯 | 870 个训练支持的稀疏点，固定拓扑，无增密 |
| RGB 初始化 | 3,000 步 Adam，更新中心/尺度/opacity/SH0 |
| 热头初始化 | 500 步 Adam，仅热头，随后销毁该 optimizer |
| CTG V2 | 100 次事务，RGB 参数冻结；每次 source + 最近合格训练 witness，source/witness 两个 RGB guards |
| anchor | 热细化前对全部训练 guards 建立固定锚点 |
| 设备 | RTX 4090；PyTorch allocated 约 50 MiB，进程总显存含 CUDA context 更高 |
| 完成时间 | 脚本计时约 29.87 s，不含下载与启动；870 个高斯的小规模开发试验 |

RGB warmup 使用去畸变有效域和接近数字白点的非饱和区域，包含极暗及较亮像素。RGB guard 单独使用训练 p99 单位下 mean>0.01、max<2 的固定可靠区域；阈值2不是传感器饱和阈值。

首次运行误将 guard 遮罩也用于 RGB warmup，属于 masked-range pilot；100 次候选全部被拒绝。独立审查发现问题后已拆分遮罩并重跑相同预算。旧日志保留为 `results/initial_masked_progress.jsonl`，不是最终结果。

## 实际结果：尚无涨点

最终 100 次候选：**1 次接受，90 次被 RGB 保护拒绝，9 次被独立热视角拒绝**。没有为了提高接受率而放松保护阈值。

| 保留视角 | RGB 初始化 MSE（分发 RAW / 65535） | 热细化后 MSE | 方向 |
|---|---:|---:|---|
| frame_00013 | 0.00018533269 | 0.00018533713 | 略差 |
| frame_00015 | 0.00032325026 | 0.00032333336 | 略差 |

训练 p99 单位下的固定可靠区域 MSE 同样略差：frame13 从 0.00447648 到 0.00449841；frame15 从 0.00230766 到 0.00244001。全图平均 PSNR 变化约 **−0.00061 dB**；变化极小，不构成收益。JSON 中 scaled-RAW PSNR 的负值来自亮灯值超出 p99 归一化后的1，不能把它当作规范化显示图的 PSNR。

这说明真实数据入口和优化已运行，但本场景上热候选通常与现有 RGB 解冲突，当前 V2 没有带来改善。可能因素包括对齐近似、稀疏固定拓扑、热外观模型、RAW 噪声和几何候选尺度；本轮不能区分原因。没有凭这个单场景结果继续扩大预算或声称算法有效。

## 产物与复现

- `rgb_checkpoint.pt`：真实 RGB 初始化及其 optimizer。
- `prefit_checkpoint.pt`：热头拟合完成的基线。
- `refined_checkpoint.pt`：100 次事务后的模型与 engine 状态。
- `audit.json` / `progress.jsonl` / `baseline_eval.json` / `final_eval.json`：数据声明、配置、源码哈希、训练/评估记录。
- `source_snapshot.py`：最终运行入口的源码副本。原始 stderr 为空。

```powershell
$env:PATH = 'E:\4D_Gaussian\.conda\nerficg\Library\bin;' + $env:PATH
& ./.conda/nerficg/python.exe -m experiments.thermal_supervision.mvtv_pilot --scene dataset/rgb_thermal/mvtv/vs/roadraw --out output/thermal_supervision/mvtv_road_new_run --allow-author-proxy --rgb-steps 3000 --thermal-steps 500 --transactions 100 --width 284 --points 12000
```

输出目录必须不存在。`--allow-author-proxy` 显式声明使用数据集投影近似；不是认证物理标定。当前入口从头运行，不提供长训练断点续跑 CLI；检查点可以由 ThermalModel 加载用于审查/后续细化。没有完整 FasterGS4D Trainer、增密、动态残差或长期收敛保证。

下一阶段应先检查真实跨谱对齐、初始化覆盖和候选拒绝分布，再做同预算直接 thermal loss / RGB-only 对照；当前结果不支持大规模基准或新颖性主张。
