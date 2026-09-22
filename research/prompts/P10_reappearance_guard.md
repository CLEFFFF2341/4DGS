# P10 消失后重现的事件保护诊断 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **B**。只看全时均值或峰值可能漏掉同一轨迹两段可见事件之间的保护需求；硬事件约束是否改善重现后的尾部误差。
最近工作：USPLAT4D遮挡；TC3DGS时序mask；SafeguardGS pixel保护。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/UNCERTAINTY MATTERS IN DYNAMIC GAUSSIAN.pdf，p4 §4.1；Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

用K1每点s_it判active= s_it≥0.05*max_t s_it且s_it>0，按24索引连续段，gap至少2个采样时刻分开，至少两段的ID为reappearing代理。它不是对象身份真值。对每个此ID的每段生成需求e=(ID,episode)。可满足该需求的候选为该ID及其K4同kind16近邻：必须在episode的每个样本t都满足s_jt>0且投影中心到i在C4中至少一相机距离≤同图8pixel，并且DC RGB距离≤.1；否则不许宣称替代。

R0从P02集合起，按episode的sum e_it降序取前min(256,floor(.1*K))需求做保护repair。若某需求无当前存活候选，插入其候选中E最高者，同时移除同kind保留集中E最低且移除不使此前已满足需求失效者；没有可移点则记infeasible，不超K。不允许后来的repair破坏先前保护。R1完全相同需求清单/成本但不repair，返回P02。

记录被保护需求数量、infeasible、回归ID重现后首个采样时间的真实V误差；固定窗口W若没覆盖重现，另输出最多5个最早重现时刻±2帧诊断（最多50图，计成本，不能替换标准指标）。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p10.py`、`configs/research/p10.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K1,K2,K4**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：两段可见且中间遮挡toy；单持续段不计重现；零分不生成需求；ID变动不让轨迹身份漂移；repair逐步验证此前constraints和K；无候选不回退随便保点。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，2行2run，0FT；若没有≥10个可行重现需求，只报事件不足，跳正式P10并继续其他任务，不人为挑test事件。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：训练相机遮挡不能推断所有视角；24采样漏短事件、轨迹对应物体不保证，可能只是漂移/透明度变化。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

与P02/P03峰值/P04比较；只有被保护事件区域重现误差下降且全局门槛过才支持。若max已足够，机制复杂性没有价值。此为B诊断，不把新约束名称包装原创。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：训练相机遮挡不能推断所有视角；24采样漏短事件、轨迹对应物体不保证，可能只是漂移/透明度变化。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
