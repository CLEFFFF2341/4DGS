# P18 受切换约束的时间活动集合 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。整轨迹删除太粗，每帧独立停用可能闪烁；区间选择是否能在等活动成本下兼顾短事件和稳定性。
最近工作：TC3DGS temporal masks；PD-4DGS；4DGC显露补偿。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7 Eq5/6；Papers/PD-4DGS Progressive Decomposition of 4D Gaussian.pdf，摘要；shared_protocol S5。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

将0..299按最近T24时间分24个整数Voronoi bins，tie归较早bin，长度n_b。z_ib∈{0,1}，目标sum_i,b n_b*e_ib*(1-z_ib)+lambda*sum_i,b n_b*z_ib+gamma*sum_i,b>0|z_ib-z_i,b-1|。活动Gaussian-frame cap=.5*N*300。R0 gamma=.1*median_{e_ib>0}(n_b*e_ib)，全0取0；R1 gamma=0，其余一致。

每点用两状态DP精确解给定lambda的链；lambda在[0,max e+2gamma/min n_b]二分20次，分别对static/dynamic原比例cap，保存最大活动数的feasible解，ties代理失真较小；全0inactive，DP ties优先inactive再少切换。这是Lagrange relaxation近似，若预算利用率<95%标budget_gap，不超预算追质量。

实体存全部N和RLE区间，渲染每t在raster前gather active，保持kind/ID map。峰值显存可能不减，磁盘可能增；测gather后的端到端延迟。hard切换不加额外fade；S5时间评测必须执行。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p18.py`、`configs/research/p18.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2,K6**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：N≤3,T≤5穷举DP目标；0/299覆盖；RLE一致；gamma0与独立threshold一致；allactive render一致；concat ID不乱。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，seed0，2行2run，实体N不变，active-frame≤.5N*300，0FT；V×T24+W及最早5个switch±2帧诊断，不替换标准指标。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **中**；风险：粗bin proxy/hard边界/DP CPU成本；保存全部点不省显存。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

同活动预算R0比R1 TDE降≥10%、平均质量门槛过；实际端到端加速才有部署主张。raster快但gather后慢则部署失败。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：粗bin proxy/hard边界/DP CPU成本；保存全部点不省显存。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
