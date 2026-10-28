# 相关论文与新颖性边界

检索截至 2026-10-29。以下只使用原始论文和作者项目；“相近”不等于复现或性能比较已完成。每项概述保持简短，所有性能主张以原论文实验范围为限。

| 原始来源 | 已有工作与本方案的关系 |
|---|---|
| [Xu et al., Leveraging Thermal Modality to Enhance Reconstruction in Low-Light Conditions, ECCV 2024](https://arxiv.org/abs/2403.14053)；[作者仓库/MVTV](https://github.com/XuJiacong/Thermal-NeRF) | 最直接任务近邻：thermal+RAW改善暗光可见重建，补充材料已有 Thermal-GS。不能声称“首次热红外监督改善暗光 GS”。MVTV固定单应对齐有视差局限，不能默认精确共光心。作者仓库本次仍标实现待更新，数据有官方链接。 |
| [ThermalGaussian](https://arxiv.org/abs/2409.07200)；[项目](https://thermalgaussian.github.io/) | 已有RGB-T高斯、标定、热平滑与多模态正则，发布RGBT-Scenes。双头及共享结构不是本方案创新。 |
| [ThermoSplat, v3, 2026](https://arxiv.org/abs/2601.15897) | 已有热结构调制和模态相关几何/opacity解耦。必须验证本方法在玻璃/反射场景是否反而不如它。 |
| [Dynamic Thermal Gaussians, 2026-09-21](https://arxiv.org/abs/2609.24531)；[DTG代码/数据](https://github.com/LinLif1869/DTG) | 已有动态RGB-T共享几何、模态形变与高斯路由；提供12组动态双视角序列。其同步和共享位姿有近似，数据需实际审计。4D扩展本身不能作为新意。 |
| [Counterfactual Route Optimization for Gaussian Head Avatar Modeling, 2026-10-07](https://arxiv.org/abs/2610.09791) | 最强方法论近邻：同状态G/A/joint临时更新、实测utility与路由控制。我们借鉴候选试探思路；潜在区别仅为跨谱独立观测的源拟合匹配和迁移检验，不宣称新通用反事实优化框架。须做其机制的热场适配对照，且标注非作者原生实验。 |
| [ThermalNeRF: Thermal Radiance Fields, Lin et al.](https://arxiv.org/abs/2407.15337) | 研究RGB/LWIR联合辐射场、两相机标定及跨谱可见性差异。提醒共几何不意味着同透明度或同像素。 |
| [Thermal-NeRF: Neural Radiance Fields from an Infrared Camera, Ye et al.](https://arxiv.org/abs/2403.10340) | 热成像重建近邻；与Xu的ECCV暗光论文不是同一篇，名称容易混淆。 |
| [TRNeRF, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Carmichael_TRNeRF_Restoring_Blurry_Rolling_Shutter_and_Noisy_Thermal_Images_with_WACV_2025_paper.pdf) | 热成像也受噪声、模糊及rolling shutter影响；禁止假定热图天然清晰。 |
| [Deblur4DGS](https://arxiv.org/abs/2412.06424)；[MoBGS](https://arxiv.org/abs/2504.15122) | 动态去模糊重建、连续运动与曝光建模均已有工作。候选的增益需超过相同曝光模型的RGB-only控制，而非只比较错误的中点渲染。 |
| [BAGS, ECCV 2024](https://arxiv.org/abs/2403.04926) | 用图像模糊建模改善GS；静态模糊场景的必要相关基线。 |
| [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224)；[PCGrad](https://arxiv.org/abs/2001.06782) | 辅助任务门控和避免梯度冲突已有先例，必须与便宜的固定权重/梯度夹角控制比较。 |
| [Ren et al., Learning to Reweight Examples, ICML2018](https://proceedings.mlr.press/v80/ren18a.html)；[HydaLearn](https://arxiv.org/abs/2008.11643) | 根据更新后的主任务表现调权重也不是新概念。当前有限步前向检验无需二阶导，但“无需二阶”只属可实施优势。 |

## 待证实的差异

把“热残差是否应该改几何”视为受传感器模型约束的选择问题：G/A源拟合下降匹配后，在不同训练视角比较迁移收益，并检查RGB实际曝光观测。目标是减少热外观变化/错误配准引发的几何损坏。该组合截至本次检索未被确认为相同已有实现，但检索不能证明全球首创；多项组件已有充分先例。

若只比RGB-only涨点，而未优于直接thermal联合训练及上述路由控制，则证据只支持“用了额外信息”，不支持新算法的独立贡献。
