# P15 自适应轨迹关键帧迁移 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。统一关键帧密度在平缓轨迹浪费字节；误差驱动分配可能保急转弯而减少静缓区存储。
最近工作：TC3DGS keypoint interpolation；Ex4DGS keyframes。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Temporally Compressed 3D Gaussian.pdf，p8 Alg1；Papers/Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf，p5 Eq6–9。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

原P_i∈R^(J×3),Q_i∈R^(J×4)，J由PLY读取含padding。新文件保存per-ID joint knot indices、P/Q，端点强制；仅压动态position/rotation。解码缺失P线性插值、Q shortest-arc slerp，恢复原J dense keyframes，再调用原cube/slerp renderer。Q先normalize，相邻dot<0取反；全保留模式不改原tensor。

R0每点首末knots起步，缺失k的epsilon_ik=||P_ik-Pinterp_ik||²/L²+(angle(Q_ik,Qinterp_ik)/pi)²。每次将全场最大epsilon的k加入对应点，更新该点误差；ties ID,k。仅在actual bytes能容纳新k时加入，达到目标或全误差0终止。R1 uniform：全场轮转按ID，逐点最长时间区间二分添中点（并列较早区间），同byte目标。索引uint16（J>65535则uint32）及每点offset计入bytes。

这是TC3DGS思想迁移，两误差项等权为固定经验选择。decode dense只省磁盘，不预称GPU/速度收益。原cube可能放大knots误差，需测过渡帧。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p15.py`、`configs/research/p15.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0,K4**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：线性/常旋转只需端点；q和-q等价；180度slerp核验；padding knots范围；allknots decode一致；t=.5真实render。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，seed0，2行2run，N不变，目标B_fixed+.5B_variable，B_variable=动态P/Q payload；0FT；128轨迹CPU+2帧GPU smoke。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：不规则索引/外推padding/quat约定；decode dense不省GPU。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

same-byte胜uniform和P01/P02；仅knots MSE改善而image恶化则代理不充分。A迁移，不声称新RDP。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：不规则索引/外推padding/quat约定；decode dense不省GPU。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
