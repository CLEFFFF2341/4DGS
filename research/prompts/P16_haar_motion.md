# P16 Haar轨迹压缩迁移 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。相邻关键帧有低频冗余；压缩位置细节是否比删点更划算。
最近工作：RD4DGS §3.3。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Temporal Smoothness-Aware Rate-Distortion.pdf，p5 Eq6/7。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

仅压动态P[N_d,J,3]，其余不变。J奇数时repeat末keyframe，记原J。一级Haar a_k=(P_2k+P_2k+1)/sqrt2，d_k=(P_2k-P_2k+1)/sqrt2。R0丢全部d，只存a float32，decode每对P=a/sqrt2，截回J再原renderer。R1直接P存FP16、decode FP32（相近position bytes，无低通）。不加quant/FT帮助R0。

两行bytes由编码固定，padding略有差异；按actual bytes报告，差>1%不称等字节。P01/P02按更大actual byte cap匹配（S4共享4个资源对照槽）。全保留模式同时保a,d并逆变换验证roundoff，不能将真正压缩误差归为数值噪声。

丢高频可能带来运动相位偏移。完整RD4DGS还含联合优化，本模块负结果不否定原论文。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p16.py`、`configs/research/p16.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：正逆Haar、常值d=0、线性轨迹pair averaging非exact、奇数长度；FP16溢出报失败不clip；位置与位移不得混用。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，seed0，2行2run，N不变，固定一级Haar/FP16，0FT；B_variable=动态P payload，约减半。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：高频运动/position精度/边界；仅磁盘收益，不能引用完整RD4DGS倍数。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

比较R1、P15、same-byte剪枝；FP16解释收益则Haar无必要；仅训练能恢复留阶段3统一预算验证。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：高频运动/position精度/边界；仅磁盘收益，不能引用完整RD4DGS倍数。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
