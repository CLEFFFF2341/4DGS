# P06 最差时间段约束的预算分配 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。平均最优的集合可能系统牺牲某个时间段；显式提高最差段保留效用能够减少尾部质量崩溃，而不是只保单点峰值。
最近工作：Light4GS时间汇总；TC3DGS时间一致性；robust allocation。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Light4GS Lightweight Compact 4D Gaussian.pdf，p5 Eq9；Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7；research/literature_matrix.md关键推导1。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

将24个时间按索引连续分为6段，每段4时刻。E_ib=sum_(t∈b) e_it；z_b(S)=sum_(i∈S)E_ib/(sum_i E_ib+1e-12)，分母为0的段从目标去掉。目标max_S min_b z_b(S)，同组配额。此min一般非次模，不能套P04保证。

R0：multiplicative weights固定5轮近似。lambda_b=1/B；每轮score_i=sum_b lambda_b E_ib/(total_b+eps)，按组topK；记录S_j、z；lambda_b←lambda_b*exp(2*(1-z_b(S_j)))再sum归一。最终从5个候选中选min_b z最大者，tie选mean z再ID。这是代理内选择，不看V；5轮只最终一个正式run。R1：lambda均匀的一轮topK，同样归一化，对照“仅分段归一化”。

观察每段贡献保留率与V上分段ΔMSE是否匹配；另外比较P03峰值法。全零时返回P01.R2。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p06.py`、`configs/research/p06.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：六段极不平衡toy应重新分配；指数稳定（先减max loglambda）；统计总和与全点z≈1；保存五轮候选及objective，最终选择不读V；检查逐组配额。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，R0/R1两行2run，0FT；5轮不渲染，禁止每轮V挑选；额外窗口W。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：5轮无全局解保证；以全模E固定处理忽略交互；min目标可过拟合最难段。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

需提升真实最差段/最坏10%且不过度伤mean（S7）；R1/P03同样好则robust约束没有独立贡献。代理z与真实段误差无关先判PROXY_INVALID。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：5轮无全局解保证；以全模E固定处理忽略交互；min目标可过拟合最难段。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
