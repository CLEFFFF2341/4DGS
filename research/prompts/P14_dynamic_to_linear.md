# P14 动态轨迹向线性静态表示回收 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **C**。训练时归为dynamic的点在最终模型中可能仅需线性位移与固定旋转；以误差证据回收轨迹存储可能比删整点安全。
最近工作：Ex4DGS分解；Swift4D / CDGS静动态分配。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf，p4 Eq5、pp5–6 opacity；scene/c_gaussian_model.py:get_static_xyz_at_t。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

对每dynamic在所有原keyframe真实时间（含padding）LS拟合x(t)=a+(t/300)b。rotation取t149 normalized quaternion，scale/SH保持。转换候选要求全部300整数时间opacity envelope range≤.01且min≥.99，rotation角距离max 2acos(abs(dot(q(t),q149)))≤2度。位置误差R_i=mean_(T24) s_it*||x_i(t)-xlinear(t)||²/(mean_t s_it+1e-12)/L²。

R0按R_i/bytes_saved升序转换；R1相同候选按max_t||x_i(t)-mean_t x_i(t)||/L（运动幅度）升序。停止于表示赛道目标或候选耗尽，实体N不变。转换_xyz=a,_xyz_disp=b,rotation=q149,opacity=base；不把b设0。实际saved含kind属性和ID元数据，不假设固定1KB。

不为达预算越过opacity/rotation gate；未达输出可达RD点及BLOCKED_TARGET。LS几何精度不保证image精度。新颖潜力仅在有证据的posthoc转换决策，静动态分解本身已知。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p14.py`、`configs/research/p14.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0,K1,K4**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：原static轨迹采样应LS精确恢复；非线性往返净位移0不能当静止；短opacity拒绝；padding真实时间；转换后kind变而parent ID保留。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，seed0，2行2run，N不变，目标B_fixed+.5B_variable（S4），0FT；B_variable=全动态比转换为static多出的payload。

阶段1配置行上限 **2**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：opacity/rotation短暂变化、position投影敏感；微小几何误差不等小图像误差。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

需胜同bytes删点且好于简单幅度排序；极少可转换则记录已充分分离，不降低保护门槛。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：opacity/rotation短暂变化、position投影敏感；微小几何误差不等小图像误差。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
