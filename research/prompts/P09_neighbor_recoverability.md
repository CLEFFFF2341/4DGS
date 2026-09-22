# P09 邻居补偿可恢复性选择 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。当前删除误差高不代表微调后不可恢复；若误差落在存活邻居颜色可表达空间，可能安全释放容量。
最近工作：MaskGaussian补偿动机；PUP Fisher；AAAI Plug-and-Play detail compensation。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/MaskGaussian Adaptive 3D Gaussian Representation from Probabilistic Masks.pdf，p3动机/p4 §3.2；research/literature_matrix.md推导4、补充检索（AAAI仅摘要，不假装复现）。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

以P02 S0为固定基线，每kind取边界两侧各256点构成Q（≤512），固定保留S_anchor=S0\Q。对i∈Q，找S_anchor内t149位置最近8点J_i；只允许保留锚点作补偿，避免循环“互相可替代”。以16-ray patch样本的实际d_i RGB向量为target，构造颜色DC Jacobian列：D_j,r=w_jr*SH_DC_const（按仓库SH2RGB导数），RGB三个通道独立，r覆盖i或邻居支持，训练ray按hash奇偶分fit/check。

每channel解ridge x=(DᵀD+lambda I)^-1 Dᵀd_i，lambda=1e-3*trace(DᵀD)/8+1e-10，把补偿引起的RGB DC变化clip[-.05,.05]后重新计算留出ray残差。R_i=mean_check||d_i-D x||²；无邻居则R_i=E_i。R0用R排序Q选足该组原Q保留数，anchors保持；R1在同Q用E排序。选择结果阶段1不实际施加x（避免偷加优化）；另做16个按原E分位等距取的候选删点+邻居DC解补偿probe，与相同20步邻居DC Adam真修复对照，仅记录恢复相关性，不用于选择。

P09预测未来有限FT可恢复性，阶段1零FT可能差；若预测经probe验证，允许以机制证据晋级，不以零FT更差直接否定。进入阶段3必须统一总1000步，不能额外给P09局部训练。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p09.py`、`configs/research/p09.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2,K3,K4,K5**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：线性可表达与正交target toy；fit/check严格独立；DC Jacobian有限差分；对补偿point最终确实保留作断言；ridge病态用float64 Cholesky，失败加10倍lambda仅一次记日志，否则该i回E；不能用平方score拟合有符号误差。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，R0/R1两行2run，0FT；16候选×最多20步probe合计≤320步、10 GPU分钟，计P09总预算；不能全场逐点重训。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **中**；风险：只估颜色恢复，不代表几何/轨迹恢复；clamp/alpha nonlinearity、ray subsampling与遮挡变化可破坏线性近似。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

主证据为check residual对真实20步恢复后的MSE排序Spearman≥.5且好于原E至少.1（决策规则）；阶段3再看最终质量。若仅近邻距离/原E能解释，降格。优化失败与线性代理不成立分开。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：只估颜色恢复，不代表几何/轨迹恢复；clamp/alpha nonlinearity、ray subsampling与遮挡变化可破坏线性近似。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
