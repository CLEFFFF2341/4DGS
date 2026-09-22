# P04 饱和时空覆盖集合选择 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。独立排序会把预算耗在同一时空patch，留下局部空洞；集合边际覆盖可以在相同K下补足。观察覆盖缺口是否预示真实新增误差。
最近工作：SafeguardGS / Mini-Splatting；Light4GS；经典最大覆盖。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4 §3.2；research/sources/Lazier_Than_Lazy_Greedy.pdf，p3 Alg1/Theorem1；research/literature_matrix.md关键推导1。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

A_ui、patch权重w_u按S3。tau_u=0.5*sum_i A_ui；sum=0的行删除。F(S)=sum_u w_u min(1,sum_(i∈S) A_ui/tau_u)。理论条件A≥0、tau固定、w≥0，F为单调次模。真实渲染不是F。

两个行：R0=饱和F；R1=相同A/tau但不饱和的modular score sum_u w_u*A_ui/tau_u。R1用来检验收益是否只是patch归一化。tau=.5是固定经验设计，不进行阈值网格。

组配额：先在static内用Kd=0目标选Ks，再在dynamic内以已选static为底集选Kd；另用微型oracle评估次序影响。每组采用stochastic greedy epsilon=.05，每一步从剩余候选均匀无放回采样ceil((N_group/K_group)*log(20))个，按真实F边际选一个，ties按ID，selected永不重复。K=0跳组。报告实际F/evals；剩余边际全0时按P01.R2补足该组预算。此组顺序约束下不声称整体1-1/e保证；该界只适用于每个固定底集的单组基数子问题。缓存CSR，不枚举N×U。

伪代码：coverage=0→for group→for k→sample IDs→delta from their sparse columns→argmax→update coverage。不得用一次性singleton排序冒充greedy。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p04.py`、`configs/research/p04.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K3**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：N≤12穷举最优集合、验证边际递减及lazy/stochastic目标值；两个同patch高分点与独占patch点例子必须可改变决策；tau为0处理；组合保留全点时一致；比较16/64ray的目标值与真实render偏差。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，2行2run，0FT；先1%点最多1000次增量smoke测CPU代价，再全规模。超过45min selection保存部分进度并FAILED_COST，不换成topK蒙混。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **中，CSR上限8GB**；风险：稀疏采样miss、冻结T、饱和阈值、组顺序；理论近似界与PSNR无关系。全量greedy的墙钟可能高于渲染节省。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

需胜P01最佳/P02/P03及R1，且覆盖缺口与真实ΔMSE有正相关；如仅F提升而image/tail不变，标PROXY_INVALID或NEGATIVE。简单归一化解释全部收益则降格工程基线。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：稀疏采样miss、冻结T、饱和阈值、组顺序；理论近似界与PSNR无关系。全量greedy的墙钟可能高于渲染节省。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
