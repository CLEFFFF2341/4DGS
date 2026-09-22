# P07 删点后重估的因果对照 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。删去遮挡点后剩余点的重要性变化，一次性排名过时；逐轮重估是否解释复杂集合方法的收益。
最近工作：GaussianPOP iterative re-quantification；PUP多轮。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/GaussianPOP Principled Simplification Framework for.pdf，§3.4（执行只需重新阅读该节）；Papers/PUP 3D-GS Principled Uncertainty Pruning.pdf，p4 §4.4。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

分4轮，从原组数n_g到k_g，轮j目标n_gj=n_g-floor(j*(n_g-k_g)/4)，j=1..4。R0每轮在当前模型重新计算P02.E，再删最低到目标；R1用原始E进行相同分批删除，且每轮也重算E用于诊断但不用于决策。这样两行统计成本相同。无中途FT，无随机重采样，C4/T24/尺寸完全相同。

每次缓存以parent model hash隔离，参考M0图像持续作为teacher质量对照，但单删E的teacher是当前轮模型。保存每轮E在共同ID上的Spearman、排名变化、实际新暴露ray数。

伪代码：M=M0; for j: collect K2(M); select using fresh/stale; apply; save intermediate; 最后eval。不把四轮当四个研究方案。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p07.py`、`configs/research/p07.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K2，每轮失效重建**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：R1最终IDs必须与P02单轮完全相同；不同轮hash不同；新暴露toy中第二点贡献应提升；没有隐性微调、opacity reset或补点；每轮原始ID保持。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，2行2run，0FT；每行4次统计计入45min run上限。先2帧4轮smoke，如统计估计超预算只标成本阻塞，不改两轮后称同法。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **中**；风险：统计成本可能主导；多轮独立误差仍非联合最优，不能引用原文含FT收益到本无FT实验。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

胜P02且新的可见性变化预测改善可支持重估；若P04不胜重估，应将其定位为省统计成本近似而非更强机制。首次崩溃先核查早停replay和cache。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：统计成本可能主导；多轮独立误差仍非联合最优，不能引用原文含FT收益到本无FT实验。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
