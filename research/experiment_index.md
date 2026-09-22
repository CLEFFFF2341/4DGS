# 实验索引

20个方案；A=基线/迁移，B=诊断，C=待证实候选。P00/P99不计入。公式、文献范围、最小矩阵和证伪条件见各prompt。显存均为未实测风险。

| ID | 方案名称 | 类型 | 核心假设 | 最近相关工作 | 潜在贡献 | 实现成本 | 预计显存风险 | 依赖缓存 | 验证顺序 | 停止条件 |
|---|---|---|---|---|---|---|---|---|---|---|
| [P01](prompts/P01_score_baselines.md) | 可复用标量与采样基线 | A | 若复杂方法只是偏好高贡献或改变采样分布，简单打分应解释大部分收益 | LightGaussian / Light4GS / Mini-Splatting / RadSplat | 迁移强基线 | 低 | 低 | K0,K1 | 1 | 基线必须保留；统计错误停依赖 |
| [P02](prompts/P02_counterfactual_error.md) | 单删除反事实误差迁移 | A | 强贡献高斯若颜色与后景相似，其真实删点误差仍可很小。观察w排名和删除失真排名错位，检验颜色/遮挡信息是否优于P01。 | GaussianPOP；Speedy-Splat sensitivity | 迁移强基线 | 中 | 中 | K2 | 2 | 单删验证不符停K2 |
| [P03](prompts/P03_temporal_aggregation.md) | 时间聚合与短暂事件诊断 | B | 时间平均会稀释短暂但不可替代的作用 | TC3DGS平均mask；RadSplat最大贡献；Light4GS时间累积 | 隔离机制与简单解释 | 低 | 低 | K2 | 3 | 无短事件证据或简单聚合足够 |
| [P04](prompts/P04_saturating_coverage.md) | 饱和时空覆盖集合选择 | C | 独立排序会把预算耗在同一时空patch，留下局部空洞 | SafeguardGS / Mini-Splatting；Light4GS；经典最大覆盖 | 可能的机制增量，未证明原创 | 中，CSR上限8GB | 中 | K3 | 4 | F提升不改善真实误差/超时 |
| [P05](prompts/P05_trajectory_prototypes.md) | 轨迹原型代表性选择 | B | 具有近似运动、形状与颜色的完整轨迹形成重复簇，选择代表点可能优于按patch质量分配预算 | OMG4聚类；GHAP；设施选址 | 隔离机制与简单解释 | 低 | 低 | K0,K1,K4 | 5 | 代表性无图像收益 |
| [P06](prompts/P06_worst_time_allocation.md) | 最差时间段约束的预算分配 | C | 平均最优的集合可能系统牺牲某个时间段 | Light4GS时间汇总；TC3DGS时间一致性；robust allocation | 可能的机制增量，未证明原创 | 低 | 低 | K2 | 6 | 最差段无改善/mean损伤超标 |
| [P07](prompts/P07_iterative_recalibration.md) | 删点后重估的因果对照 | A | 删去遮挡点后剩余点的重要性变化，一次性排名过时 | GaussianPOP iterative re-quantification；PUP多轮 | 迁移强基线 | 中 | 中 | K2，每轮失效重建 | 7 | 重估无排名/质量收益或统计超时 |
| [P08](prompts/P08_joint_deletion.md) | 联合删除的颜色与遮挡交互 | C | 两个各自安全的删除组合可能不安全，或者其颜色误差互相抵消 | GaussianPOP单删；PUP忽略跨点block；MaskGaussian自适应 | 可能的机制增量，未证明原创 | 高，严格chunk | 高 | K2,K3,K5 | 8 | 交互弱/交换无真实收益 |
| [P09](prompts/P09_neighbor_recoverability.md) | 邻居补偿可恢复性选择 | C | 当前删除误差高不代表微调后不可恢复 | MaskGaussian补偿动机；PUP Fisher；AAAI Plug-and-Play detail compensation | 可能的机制增量，未证明原创 | 中 | 中 | K2,K3,K4,K5 | 9 | 恢复预测不优于原E |
| [P10](prompts/P10_reappearance_guard.md) | 消失后重现的事件保护诊断 | B | 只看全时均值或峰值可能漏掉同一轨迹两段可见事件之间的保护需求 | USPLAT4D遮挡；TC3DGS时序mask；SafeguardGS pixel保护 | 隔离机制与简单解释 | 低 | 低 | K1,K2,K4 | 10 | 重现需求少于10/峰值已解释 |
| [P11](prompts/P11_learned_budget_gate.md) | 可恢复的定额掩码优化迁移 | A | 在删除前优化集合参与度可让相关点竞争与恢复，比固定评分更接近重建目标 | MaskGaussian / GaussianSpa / CDGS / RD4DGS | 迁移强基线 | 高 | 高 | K2,K6，训练后全失效 | 11 | hard坍塌/梯度错误/不胜等步FT |
| [P12](prompts/P12_static_dynamic_allocation.md) | 静动态容量分配诊断 | B | 原始静动态数量比例可能不是压缩后的最佳比例 | CDGS adaptive allocation；Ex4DGS / Swift4D | 隔离机制与简单解释 | 低 | 低 | K2 | 12 | 收益仅来自更多昂贵动态容量 |
| [P13](prompts/P13_trajectory_merge.md) | 轨迹一致的矩匹配合并迁移 | A | 直接删除丢掉局部质量，两个近似轨迹合为一个能保留位置/形状信息 | OMG4 Gaussian Merging；GHAP mixture reduction | 迁移强基线 | 中 | 中 | K0,K2,K4 | 13 | 合并机会<1%N/不胜删父点 |
| [P14](prompts/P14_dynamic_to_linear.md) | 动态轨迹向线性静态表示回收 | C | 训练时归为dynamic的点在最终模型中可能仅需线性位移与固定旋转 | Ex4DGS分解；Swift4D / CDGS静动态分配 | 可能的机制增量，未证明原创 | 低 | 低 | K0,K1,K4 | 14 | 合格转换不足/不胜小运动控制 |
| [P15](prompts/P15_adaptive_keyframes.md) | 自适应轨迹关键帧迁移 | A | 统一关键帧密度在平缓轨迹浪费字节 | TC3DGS keypoint interpolation；Ex4DGS keyframes | 迁移强基线 | 低 | 低 | K0,K4 | 15 | 不胜uniform/解码错误 |
| [P16](prompts/P16_haar_motion.md) | Haar轨迹压缩迁移 | A | 相邻关键帧有低频冗余 | RD4DGS §3.3 | 迁移强基线 | 低 | 低 | K0 | 16 | 不胜FP16/运动失真 |
| [P17](prompts/P17_shared_motion_basis.md) | 跨轨迹共享低秩运动基 | A | 同物体轨迹可能共用运动基，跨点相关压缩能否胜逐轨迹时间编码。 | 4DGC共享motion grid；TC3DGS / RD4DGS | 迁移强基线 | 低 | 低 | K0,K4 | 17 | 不胜独立DCT |
| [P18](prompts/P18_temporal_active_set.md) | 受切换约束的时间活动集合 | C | 整轨迹删除太粗，每帧独立停用可能闪烁 | TC3DGS temporal masks；PD-4DGS；4DGC显露补偿 | 可能的机制增量，未证明原创 | 中 | 中 | K2,K6 | 18 | TDE不降/端到端不加速 |
| [P19](prompts/P19_resource_budget.md) | 按实际资源成本选择 | C | 动态轨迹贵、大投影点慢 | LightGaussian / Speedy-Splat / RD4DGS / CDGS | 可能的机制增量，未证明原创 | 低 | 低 | K0,K1,K2 | 19 | 资源代理不对应实测/不胜同cap |
| [P20](prompts/P20_attribute_rate_allocation.md) | 按视时敏感度分配SH精度 | A | 各点高阶外观需求不同，逐点SH精度分配是否优于统一降阶。 | RD4DGS SH mask；TC3DGS mixed precision；LightGaussian | 迁移强基线 | 中 | 中 | K0,K1,K3 | 20 | 不胜uniform/高光泛化失败 |

## 阶段1运行数

| ID | 参数行上限 | 额外优化 |
|---|---:|---|
| P01 | 6 | 0FT；内部搜索/统计计时 |
| P02 | 1 | 0FT；内部搜索/统计计时 |
| P03 | 3 | 0FT；内部搜索/统计计时 |
| P04 | 2 | 0FT；内部搜索/统计计时 |
| P05 | 2 | 0FT；内部搜索/统计计时 |
| P06 | 2 | 0FT；内部搜索/统计计时 |
| P07 | 2 | 0FT；内部搜索/统计计时 |
| P08 | 2 | 0FT；内部搜索/统计计时 |
| P09 | 2 | 16候选probe≤320步，仅诊断 |
| P10 | 2 | 0FT；内部搜索/统计计时 |
| P11 | 3 | 200步，含等步控制 |
| P12 | 3 | 0FT；内部搜索/统计计时 |
| P13 | 2 | 0FT；内部搜索/统计计时 |
| P14 | 2 | 0FT；内部搜索/统计计时 |
| P15 | 2 | 0FT；内部搜索/统计计时 |
| P16 | 2 | 0FT；内部搜索/统计计时 |
| P17 | 2 | 0FT；内部搜索/统计计时 |
| P18 | 2 | 0FT；内部搜索/统计计时 |
| P19 | 3 | 0FT；内部搜索/统计计时 |
| P20 | 2 | 0FT；内部搜索/统计计时 |

方法行合计 **47**。另4个共享资源匹配对照槽=**51**。按首次需要的实际byte/tile cap建立P01/P02对照，后续同key复用；超过4种新cap则BLOCKED_BUDGET，不声称公平强对照齐全，阶段2额度内再补。共同未剪基线另最多9run。阶段2≤96、阶段3≤108，正式评测run≤264；另24次失败修复槽。见shared_protocol S7。

P01六分数、P03三聚合、P11两mask优化都是内部控制，没有独立凑方案。P04图像覆盖、P05轨迹代表性、P06最差时间段、P10硬事件保护分别有不同失败假设。P14类型转换、P15稀疏knots、P16固定小波、P17跨点共享、P20外观精度操作不同信息结构，均不重复申报新颖性。
