# P11 可恢复的定额掩码优化迁移 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。在删除前优化集合参与度可让相关点竞争与恢复，比固定评分更接近重建目标；检验收益是否仅来自额外训练。
最近工作：MaskGaussian / GaussianSpa / CDGS / RD4DGS。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/MaskGaussian Adaptive 3D Gaussian Representation from Probabilistic Masks.pdf，p4 Eq3/4；Papers/GaussianSpa An “Optimizing-Sparsifying” Simplification Framework for.pdf，p5 Eq6–16；Papers/Temporal Smoothness-Aware Rate-Distortion.pdf，p4 Eq4。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

固定原模型属性，新增每点logit l_i，初始化logit(0.9)，加1e-3*标准化P02.E打破tie，E标准化用mean/std+1e-8。组内K固定。

R0=hard topK STE：p=sigmoid(l)，h=topK(p)的0/1，gate=stopgrad(h-p)+p；在raster内部把alpha改为gate*a且保留原a参与列表，T递推使用(1-gate*a)。alpha阈值判断用未masked a，不让h=0点丢失mask梯度。全部非gate属性冻结，Adam(l,lr=.01) 200步，batch1 C随机视时、0.8L1+.2DSSIM；每步精确K但全候选仍占显存/统计时间，最终实体删除topK。

R1=soft gate budget：相同l/200步，gate=p，loss增加10*((sum_g p-K_g)/N_g)^2对两组求和；最后hard topK，记录soft→hard质量落差。R2=P01.R2直接剪到K后对全部保留属性做200步S6普通FT（相同batch/数据抽样），作为额外训练解释对照。R0/R1只是在P11中的优化器诊断，不独立计创新。

理论：STE有偏梯度；soft数量不是精确非零点数；非凸解无全局保证。不是原版MaskGaussian的Gumbel或GaussianSpa完整ADMM复现。实现mask梯度后再开运行；禁用densify等S6操作。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p11.py`、`configs/research/p11.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2,K6，训练后全失效**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：mask=all1与原render等价；mask0仍有有限差分方向一致的mask梯度；2点可替代toy验证一方masked仍可恢复；hard前向点数精确；验证opacity筛选未截断被mask点；梯度不写回冻结属性。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，3行3run；每行200步选择/训练计入预算，不标0FT方法；正式前10步smoke。阶段3 R0/R1再800步普通FT；R2从原checkpoint重做总1000步，不把已200步结果再加1000。

阶段1配置行上限 **3**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **高**；风险：掩码CUDA反传改动、筛选/早停梯度、Windows扩展编译风险最高；不能用普通opacity乘法伪装完整恢复mask。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

必须同时看P02零FT和R2等步FT，只有胜等成本控制才支持学习选择价值。soft模型好但hard坍塌属OPTIMIZATION_FAILED，不追加训练到成功。作为强迁移基线保留。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：掩码CUDA反传改动、筛选/早停梯度、Windows扩展编译风险最高；不能用普通opacity乘法伪装完整恢复mask。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
