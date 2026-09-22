# P03 时间聚合与短暂事件诊断 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **B**。时间平均会稀释短暂但不可替代的作用；比较mean、max、上尾均值，观察短可见点是否被救回。三个聚合属于一个诊断，不是三个创新。
最近工作：TC3DGS平均mask；RadSplat最大贡献；Light4GS时间累积。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7 Eq5/6；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4 score表。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

输入固定K2的e_it≥0，先每t平均视角/像素，再在24个t聚合：R0=mean_t e_it；R1=max_t e_it；R2=mean最高ceil(.1*24)=3个e_it。三者保留最高分，时间未见必须0而非缺失跳过；最大值只有一个time也允许。不得加入mean与max混合权重。

机制标签：support_i=#{t:e_it>0.05*max_t e_it}，max=0则support=0；短暂=1..3，持续≥12。报告按标签组的保留概率、被删点E分布及其投影区域新增误差。不是语义标签，不把检测到的motion ROI当人工真值。

伪代码：e[N,24]→三个row score→同Ks/Kd→比较；额外以T24奇偶各12个时间计算排名稳定性（只诊断，不产生第四选择模型）。max/CVaR是风险偏好，不能套单删平均loss最优性证明。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p03.py`、`configs/research/p03.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：构造单尖峰与恒定序列使三规则产生预期不同排序；全0不NaN；量纲相同；每视角缺帧触发manifest错误；检查采样遗漏尖峰时任何聚合都无能为力。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，3行≤3run（R0=P02若hash相同直接复用，仍占上限槽），0FT，顺序mean→max→tail；追加固定窗口W时间评测。

阶段1配置行上限 **3**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：极端max对噪声/单视角异常敏感；24点没有短事件保证。不能宣称时域稳定性仅靠低最差帧MSE。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

若P04/P10不胜这里的简单聚合，复杂机制收益可被简单解释。短可见标签无尾部质量关联时不继续构造复杂“重现保护”；先记采样aliasing或数据事件不足。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：极端max对噪声/单视角异常敏感；24点没有短事件保证。不能宣称时域稳定性仅靠低最差帧MSE。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
