# P17 跨轨迹共享低秩运动基 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。同物体轨迹可能共用运动基，跨点相关压缩能否胜逐轨迹时间编码。
最近工作：4DGC共享motion grid；TC3DGS / RD4DGS。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：research/sources/4DGC.pdf，p4 §3.1/3.2；Papers/Temporally Compressed 3D Gaussian.pdf，p8。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

只压动态P[N_d,J,3]。每点首关键帧p_i0单存，X_i=vec(P_i-p_i0)∈R^(3J)。R0 CPU分块XᵀX（3J维，禁止N²），eigh取前r个eigenvectors V，A=XV，Xhat=AVᵀ。r为使4*(3Nd+Nd*r+3J*r)+header≤target_variable_bytes的最大整数，r≤min(Nd,3J)，r<1报目标不可行。eigenvector符号按首非零分量正规范。

R1每条坐标序列正交DCT-II时间basis，保前r_t系数，r_t取满足相同actual byte cap的最大整数，三轴等长；DCT基固定解析不用存，CPU显式J×J矩阵可实现，无需安装库。rotation等不变。解码denseFP32再原renderer。

低秩理论仅保证Frobenius最佳rank-r误差，不保证图像/可见性最优；不增加exceptions残差文件突破预算。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p17.py`、`configs/research/p17.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0,K4**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：共享平移rank≤3可恢复；随机独立轨迹不应虚报低rank；full rank相对误差≤1e-5；分块Gram与小集SVD核验；大负eigenvalue报错；basis/offset成本计入。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，seed0，2行2run，N不变，B_variable=动态P，目标B_fixed+.5B_variable，0FT；1024轨迹smoke。CPU≤45min，显存估计低但未实测。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：多物体全场低秩混合/绝对坐标/decode dense；速度可能更慢。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

同bytes比较R1/P15/P16；DCT解释收益则跨轨迹共享无价值。A编码迁移，不宣称新数学工具。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：多物体全场低秩混合/绝对坐标/decode dense；速度可能更慢。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
