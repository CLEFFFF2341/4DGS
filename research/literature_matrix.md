# 文献矩阵与研究判断

截至 2026-09-22。先筛全部29份本地研究PDF，再针对最接近的方法段阅读；未读完整全文的论文不称全文精读。L0=检索条目/摘要；L1=本地摘要与引言筛查；L2=指定方法/公式细读；L3=全文含附录核对。本轮没有把任何论文标成L3。文本提取成功不等于读过全部页。页码为PDF物理页。原文在 `Papers/`，逐页提取在 `research/paper_text/`，准确文件映射见 `paper_inventory.json`。

## 动态工作与直接竞争

| 工作、来源 | 深度与已读位置 | 已确认机制 / 对本项目的约束 | 可复用性 |
|---|---|---|---|
| [Ex4DGS](https://arxiv.org/abs/2410.15629) | L2，本地PDF pp4–6，§4.1–4.4，Eq5–12；本地源码核对 | 线性静态位移、关键帧位置/旋转、时域opacity、按累积图像误差回溯剪枝；“静态”不等于零运动。 | 当前仓库/基模；先复用，不重建Wu 4DGS。 |
| [OMG4](https://arxiv.org/abs/2510.03857)、[作者页](https://minshirley.github.io/OMG4/) | L2，本地pp3–7，§4.1–4.4、Eq1–5；p6公式视觉核对 | 梯度SD score→阈值prune→4D grid内合并→外观网络/SVQ。主干是Real-Time4DGS，非Ex4DGS。独立打分、合并、量化均已有先例。 | 迁移概念可行；不能移植4D均值时间梯度为Ex4DGS轨迹梯度而仍称原法。 |
| [TC3DGS](https://arxiv.org/abs/2412.05700)、[作者页](https://ahmad-jarrar.github.io/tc-3dgs/) | L2，本地pp3–8，§4.1.1–4.1.3，Eq1–8、Alg1 | 时间掩码一致性、平均mask剪枝、梯度混合精度、自适应轨迹关键点；主干Dynamic 3DGS逐帧参数。 | P15是迁移，不是新RDP；时间一致性也不能独占。 |
| [Light4GS](https://arxiv.org/abs/2503.13948)、[官方代码](https://github.com/Evan-sudo/Light4GS) | L2，本地v2 pp3–5，§IV-B Eq9；README核对 | STP将跨时间/视角的ray hit、T、opacity、normalized volume汇总；另压HexPlane和SH。时空平均贡献不是本项目创新。 | P01 STP迁移；当前无HexPlane。README称当前代码无稳定AQ；本地v2摘要12×与检索摘要120×有版本差，不拿混用数字对标。 |
| [RD4DGS](https://arxiv.org/abs/2507.17336)、[官方代码](https://github.com/HyeongminLEE/RD4DGS) | L2，本地v3 pp3–5，§3.2–3.4 Eq4–8；README核对 | 直接基于Ex4DGS；高斯/SH mask、ECVQ、一级Haar且丢高频、RD优化。 | 最重要紧凑表示对照。官方README预训练为TBD；不能承诺可直接下载checkpoint。P16仅Haar模块迁移，不冒充完整RD4DGS。 |
| [CDGS](https://arxiv.org/html/2602.03538v1) | L2，在线§III-A/B Eq5–12及§III-C标题/概述；未完整审计代码 | 可微数量控制、motion/perceptual score、静动态预算分配；预算控制与静动态分配已有人做。 | P11/P12为迁移/诊断。PDF下载IncompleteRead失败，保留在线定位；不假称本地完整PDF。 |
| [MEGA](https://arxiv.org/abs/2410.13613)、[正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MEGA_Memory-Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes_ICCV_2025_paper.pdf) | L2，本地pp3–4，§3.1/3.2、Fig3；摘要 | 4D primitive + compact AC predictor + deformation扩大作用范围 + opacity entropy。 | 是表示级训练法，不能和posthoc剪枝只比论文压缩倍数。 |
| [4DGC](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_4DGC_Rate-Aware_4D_Gaussian_Compression_for_Efficient_Streamable_Free-Viewpoint_Video_CVPR_2025_paper.pdf)、[官方代码](https://github.com/qianghu-huber/4DGC) | L2，新下载 `sources/4DGC.pdf` pp1,3–5，§3.1–3.3 | motion grid预测+新显露区域补偿高斯+按帧RD编码。 | 流式frame bytes不能与Ex4DGS整场景bytes混比；重现/新显露是必要失败案例。 |
| [Swift4D](https://openreview.net/pdf?id=c1RhJVTPwT)、[作者代码](https://github.com/WuJH2001/swift4d) | L2，本地pp3–4，§3概述/Fig3；摘要 | 学习静动态分类，仅动态走4DHash，后续density control。 | P14反向转换仅posthoc机制试验，非新的静动态分解。 |
| [USPLAT4D](https://arxiv.org/abs/2510.12768) | L2，本地p4 §4.1 Eq2–4 | 基于观测/贡献平方估计时变不确定性，图传播可靠运动；针对单目遮挡漂移。 | 低观测不等于冗余；用于P10失败分析，不能直接认为高不确定性应删除。 |
| [PD-4DGS](https://arxiv.org/abs/2605.11427) | L1，本地摘要 | 分层motion decomposition、渐进传输、时域mask一致性。 | P18有先例边界；本项目不声称渐进streaming新颖。 |
| [CC-4DGS](https://arxiv.org/abs/2609.02184) | L1，本地摘要 | computational deformation field、canonical属性编码。 | 当前表示不同；本地首页出版信息未外部独立核验，不作已录用结论。 |
| [4C4D](https://arxiv.org/abs/2604.04063) | L1，本地摘要 | 稀疏4相机几何/外观不平衡，opacity decaying。 | 不是压缩直接对照，不为本项目追加稀疏视角训练成本。 |
| [ReconDrive](https://arxiv.org/abs/2603.07552) | L1，本地摘要 | feed-forward driving、静动态头、速度组成。 | 场景/训练范式不同；本轮不迁移大模型。 |
| [Wu 4D-GS](https://arxiv.org/abs/2310.08528) | L1，本地摘要 | HexPlane+MLP deformation。 | 用来消歧；不能把其架构写成当前项目事实。 |

## 静态压缩、剪枝与可迁移工具

| 工作、来源 | 深度与位置 | 关联、局限 |
|---|---|---|
| [GaussianPOP](https://arxiv.org/abs/2602.06830) | L2，本地pp3–4 §3.1–3.3 Eq1–4/Alg1；p3视觉核对 | 单删除真实颜色差而非贡献强度；迭代重估已有先例。P02/P07是迁移。固定全模单删误差相加不等于联合删除误差。 |
| [PUP](https://arxiv.org/abs/2406.10219)、[作者页](https://pup3dgs.github.io/) | L2，本地pp3–4 §4.1–4.4 Eq5–10 | JᵀJ近似、位置scale块logdet、patch统计、多轮prune-refine。丢弃residual Hessian是近似；当前渲染误差非零时不能说精确。 |
| [Speedy-Splat](https://arxiv.org/abs/2412.00578) | L2，本地pp4–6 §4.1/4.2 Eq17–21 | tile精确求交与scalar projected-Gaussian sensitivity，减少PUP统计成本。P19须实测tile与时间的关系。 |
| [Mini-Splatting](https://arxiv.org/abs/2403.14166) | L2，本地pp6–7,9 §4.1/4.2 | 最大贡献交点保留与importance sampling，空间分布是关键；P01包含采样，P04不能仅胜随机就宣称新颖。 |
| [SafeguardGS](https://arxiv.org/abs/2405.17793)、[官方代码](https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS) | L2，本地pp3–4 §3.1/3.2 | per-pixel union保留避免射线无交点。固定严格K时union可能超预算，不能同时许诺严格budget和所有ray safeguard。 |
| [MaskGaussian](https://arxiv.org/abs/2412.20522)、[官方代码](https://github.com/kaikai23/MaskGaussian) | L2，本地pp3–4 §3.2–3.4 Eq3–6 | 被mask点仍收mask梯度、允许恢复；P11不是首次可恢复剪枝，P09需证明局部恢复预测有额外价值。 |
| [GaussianSpa](https://arxiv.org/abs/2411.06019) | L2，本地pp3–5 §3.2/3.3 Eq5–16 | L0约束、变量分裂、优化/稀疏交替；非凸问题不获全局最优。P11明确是受启发的posthoc gate控制，不称严格复现。 |
| [LightGaussian](https://arxiv.org/abs/2311.17245) | L1，本地摘要；score通过SafeguardGS §3.1交叉核对 | significance+recovery、SH蒸馏/VQ；P01低成本基线，P20已有属性压缩先例。 |
| [Compact 3DGS](https://arxiv.org/abs/2311.13681) | L1，本地摘要 | mask+神经颜色+VQ；P11/P20必须标迁移。 |
| [LP-3DGS](https://arxiv.org/abs/2405.18784) | L1，本地摘要 | Gumbel-Sigmoid学习mask；不是新颖性空白。 |
| [RadSplat](https://arxiv.org/abs/2403.13806) | L1，本地摘要；max score由SafeguardGS §3.1核对 | max而非sum贡献是已有替代；P03不可把max当新方法。 |
| [TrimGS](https://arxiv.org/abs/2406.07499) | L1，本地摘要 | contribution trimming +小尺度几何；仅图像指标不能声称几何变好。 |
| [Trimming the Fat](https://arxiv.org/abs/2406.18214) | L1，本地摘要 | gradient-informed iterative posthoc pruning。 |
| [Taming 3DGS](https://arxiv.org/abs/2406.15643) | L1，本地摘要 | constructive budgeted densification；本项目固定checkpoint不从头重建。 |
| [SVR-GS](https://arxiv.org/abs/2509.11116) | L1，本地摘要 | per-ray空间mask regularizer；空间覆盖/局部稀疏压力已有相近动机。 |
| [GHAP](https://arxiv.org/abs/2506.09534)、[作者代码](https://github.com/DrunkenPoet/GHAP) | L1，本地摘要 | OT Gaussian mixture reduction + geometry/appearance解耦；P13 moment merging是迁移基线，不能宣称首个分布合并。 |

以上29份本地研究PDF均完成至少摘要筛查；glm manual.pdf是第三方库文档，非研究论文，未纳入。页面全文提取供以后查阅，不扩大阅读深度声明。

## 数学工具、补充检索与原创性边界

- [Nemhauser–Wolsey 1978](https://pubsonline.informs.org/doi/10.1287/moor.3.3.177)：L0，原始出版页/摘要核对；单调次模最大化的经典界。这里不声称读过其全文证明。
- [Lazier Than Lazy Greedy](https://arxiv.org/abs/1409.7938)：L2，下载 `sources/Lazier_Than_Lazy_Greedy.pdf` pp2–3，Alg1、Theorem1：随机候选数ceil((N/K)log(1/epsilon))，期望1-1/e-epsilon仅对固定、非负、单调次模目标及基数约束。设施选址/样本摘要已有应用；不是“首次用次模”。
- 2026-09-22检索组合：Gaussian splatting pruning submodular coverage selection；Gaussian pruning recoverability neighbor compensation；全部用户点名缩写；primary domains arxiv/CVF/OpenReview/作者代码。没有获得可核验的“时空覆盖次模剪枝完全同式”论文，并不证明不存在。检索出现 CoverPruneGS 的第三方条目，未获可验证主来源/全文，不据此判断优先权，保留为后续新颖性审计线索。
- [REFINE](https://arxiv.org/abs/2606.09074)、[GETA-3DGS](https://arxiv.org/abs/2605.02086)：L0，检索摘要；分别涉及render-free importance、联合结构剪枝量化。进入论文写作前补读，不能宣称“首个低成本/联合预算”。
- [Plug-and-Play Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/37222)：L0，正式条目摘要，有probabilistic pruning/detail compensation；P09的新颖性需进一步对照。

## 支撑设计的关键推导（本项目推导，不是论文原话）

1. 固定原模的A≥0，F(S)=sum_u w_u min(sum_(i∈S) A_ui/tau_u,1)，tau_u>0。边际=min(A_ui/tau_u, max(0,1-current_u))随S扩大不增，因此单调次模。结论只对这个代理成立：删除会改变T、背景、遮挡，真实image error一般非单调、非次模。P04先检验代理→真实误差相关性。
2. 单删 d_i=w_i(c_i-b_after)。若同时删除D并冻结d，近似误差||sum_D d_i||²=sum_D||d_i||²+2sum_(i<j)〈d_i,d_j〉。交叉项可能正/负；而删除会改变d本身，所以P08必须与完整重渲染校验，不能当严格二阶展开。
3. GT损失增量为||r-d||²-||r||²=||d||²-2〈r,d〉，r=I_ref-GT。teacher差较小不代表GT损失一定更低；保留这两个指标避免混用。
4. 线性邻居补偿min_x||d_i-D_J x||²+lambda||x||²，仅在颜色/opacity小扰动及同一可见性邻域成立；求解下降不保证有限步真实训练恢复。P09用留出ray与真实局部修复评估。
5. 3D Gaussian mixture的矩匹配可保质量中心/协方差，却不保有序alpha复合图像；轨迹交叉/颜色冲突会失败。P13保持动态scale时不变的Ex4DGS约束，不制造模型不支持的时变scale。
6. B0_static≈65×4 Ns，B0_dynamic≈315×4 Nd（PLY payload，另有header）。同K不同静动态分配会显著改变bytes，故P12同时报bytes，P19另设成本赛道。

## 结论与阅读限制

“时空覆盖次模剪枝”保留为P04候选，不提前选为主方向。必要否定：若P01/P02、简单尾部聚合或P07重估解释其全部收益，降格为工程近似/负结果。P08/P09/P10只有在真实交互、可恢复性、重现失败分别被观察到时才值得扩展。表示方案P14–P20各自衡量信息单位，不用更高保留数伪造优势。

最接近文献的方法段已细读，但多数完整实验/附录未全面重算。原论文报告的倍数不跨表示、分辨率、硬件直接比较。本轮没有实验支持任何收益承诺；发表前还需逐个C方案的正式新颖性审计与原始强基线完整复现。
