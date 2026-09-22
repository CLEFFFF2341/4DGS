# P13 轨迹一致的矩匹配合并迁移 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。直接删除丢掉局部质量，两个近似轨迹合为一个能保留位置/形状信息；检测何时合并比保一删一更好。
最近工作：OMG4 Gaussian Merging；GHAP mixture reduction。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING.pdf，pp6–7 Eq3–5；literature_matrix推导5。Eq3原印正色差项与文字相似性不符，本任务不复现该式。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

原M0每kind找t149位置16近邻，候选无重叠pair。距离d_ij=max_(t∈T24)||x_i(t)-x_j(t)||/L + ||RGB_DC_i-RGB_DC_j||；L=原场景bbox diagonal，零则1。按d升序/ID tie贪婪配对。只允许颜色差≤.1、max位置距离≤2*max_i,j(mean scale)，且动态opacity envelope在全部300整数时间的max abs差≤.1；不跨kind。

权重h_i=E_i/(E_i+E_j)，双0用.5。每时间mu=h_i*x_i+h_j*x_j；Sigma(t)=sum_l h_l[Σ_l(t)+(x_l-mu)(x_l-mu)^T]。兼容动态scale恒定：取T24平均Sigma做eigen分解，eigenvalue下限1e-8*L²；其scale/rotation在所有keyframe固定（会损失原旋转，须报告）。静态mean/disp做相同加权；动态position所有原keyframe组合。SH逐系数加权；base opacity=1-(1-o_i)*(1-o_j)，clip[1e-6,1-1e-6]转logit。动态envelope取E较高父点，tie ID。不能擅自发明时变scale。

R0从最相似pair依次merge到组K或无可行pair；若未达K，用P02删除最低E单点补预算并记录merge贡献数量。R1相同pair顺序，每pair只留E高父点，其余补预算相同。合并E以父E和仅作fallback排序，不称新真实贡献。无FT。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p13.py`、`configs/research/p13.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0,K2,K4**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：同位置/协方差时moment一致；Sigma PSD、旋转正交/quat规范；opacity有限；父ID不重复合并；全保留identity；ray颜色不守恒，toy展示误差。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，2行2run，0FT；32pair×2帧probe再全模型；可行pair<1%N报机会不足，不放宽阈值。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **中**；风险：透明度非线性、轨迹交叉、时变cov受限；父ID管理。引入时变scale属于新研究决策，不临场扩展。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

R0胜R1/P02且实际合并比例充分才支持；fallback剪枝收益不算合并有效。明确是迁移控制，不是完整OMG4复现。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：透明度非线性、轨迹交叉、时变cov受限；父ID管理。引入时变scale属于新研究决策，不临场扩展。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
