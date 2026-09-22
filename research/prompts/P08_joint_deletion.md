# P08 联合删除的颜色与遮挡交互 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。两个各自安全的删除组合可能不安全，或者其颜色误差互相抵消；明确联合干预应改变边界点选择。
最近工作：GaussianPOP单删；PUP忽略跨点block；MaskGaussian自适应。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/GaussianPOP Principled Simplification Framework for.pdf，p3 Eq4；Papers/PUP 3D-GS Principled Uncertainty Pruning.pdf，p4 Eq8–10；research/literature_matrix.md推导2。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

以P02组内topK为初集S。每组取保留边界最低32点与删除边界最高32点（不足取全部），共≤64候选/组，其余点固定。用K3采样ray保存原完整alpha有序列表，以移除候选及固定已删点后的真实重放计算D(S)=mean||I_S-I_M0||²，包含背景、重新算T，不能冻结原T。

R0=交互交换：最多16次交换/组。每步枚举candidate内所有(保留a,删除b)交换，分块评估同ray replay D(S-a+b)，只重放交换两点涉及的ray，其余ray的D为固定常数；选最小且严格改善>1e-10的一对；更新S并重放。每步每组最多32²交换，GPU逐batch16对，选定后验证真实2帧render（每4步一次，最多4次/组）。若预计成本超45min按原候选大小停止，不临时缩候选取得成功。R1=相同候选/迭代上限，但用sum_(deleted) E_i加性目标交换；理论上P02已最优，通常零交换；同样执行诊断replay计公平冷启动成本。

辅助报告32固定邻近候选对的I_ij=D(delete i,j)-D(delete i)-D(delete j)与signed-d内积近似；这是诊断不是新的调参。这里非submodular，无全局保证。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p08.py`、`configs/research/p08.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2,K3,K5**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：2前景同色点前后遮挡toy：独删低误差但双删高误差；正负颜色误差抵消toy；对≤8点穷举检查exchange目标；swap保持kind数量；replay对真实render误差按S3门槛。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，2行2run，0FT；64候选/组上限、16 swaps预注册；先8点/2ray CPU和2帧GPU smoke。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **高，严格chunk**；风险：组合渲染代价高，局部搜索非全局；忽略候选外可换点，结果阴性仅限边界交换模型。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

需真实joint interaction幅度显著于数值噪声且R0真实质量优于P02/R1；如果仅采样D改善或交换0，报代理不足/局部无机会，不宣称机制有效。未来扩展只由S7决定。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：组合渲染代价高，局部搜索非全局；忽略候选外可换点，结果阴性仅限边界交换模型。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
