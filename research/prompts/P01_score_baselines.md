# P01 可复用标量与采样基线 — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **A**。若复杂方法只是偏好高贡献或改变采样分布，简单打分应解释大部分收益；观察各规则的ID重叠、平均和尾部误差。只改变选择规则，不训练新表示。
最近工作：LightGaussian / Light4GS / Mini-Splatting / RadSplat。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：Papers/Light4GS Lightweight Compact 4D Gaussian.pdf，p5 §IV-B Eq9；Papers/Mini-Splatting Representing Scenes with a.pdf，p9 §4.2；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，pp3–4 §3.1。
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

变量采用S3。每种规则分别在static/dynamic组内选最高分Ks/Kd，不跨组重分配。

六个预注册行：R0=seed0均匀无放回；R1=mean_t base opacity at t（包含动态envelope，不用logit）；R2=mean_t s_it；R3=m_i=max_(v,t,p) w_ir；R4=STP迁移：mean_r 1(hit_ir)*T_ir*o_i(t)*gamma_i(t)，o是投影前激活opacity，gamma=min(volume_i/q90(volume_all),1)^0.1，volume=prod(scale)，q90=0时gamma=0；R5=用R2作无放回权重采样。

R5用exponential race：key_i=-log(u_i)/max(score_i,1e-30)，u∈(0,1)，从小选；先正分，正分不足则零分stable ID补齐；全零退化R0并标明。R4命中由真实alpha阈值后ray参与记录，不用radii。R4是明确定义的Ex4DGS迁移，gamma的q90与指数是本项目固定选择，不宣称逐行复现所有版本Light4GS。

伪代码：load M0,K1 → scores per row → per-kind select → gather all fields → serialize → S5评测。R2/R3不因多算指标增加采样。

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/p01.py`、`configs/research/p01.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**K0,K1**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：R0每组预算和无重复；R5小型3点Monte Carlo验证抽样频率随权重增加且无替换；opacity用sigmoid后值；R2以CPU逐ray累积对照；R3不是mean/max_time的误写；R4 q90仅模型与训练统计，不看V。
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

S0/M0，b=.5，seed0，六行=6run，均0FT；依次R0→R1→R2→R3→R4→R5。以后表示赛道需要的4个字节匹配对照由P99调度附属P01/P02，不额外搜索。

阶段1配置行上限 **6**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **低**；风险：只测到贡献不代表真实删除误差；R4 volume权重可能偏大点。数值原理是排序/加权采样，渲染有效性是经验假设。 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

这是必须保留的参照，不按收益淘汰。R2必须完成才能运行大部分复杂方法；如R4统计未实现，其余继续且R4标BLOCKED_CACHE。选择V最好的简单行用于阶段2，但必须保留全部六行失败/结果。

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：只测到贡献不代表真实删除误差；R4 volume权重可能偏大点。数值原理是排序/加权采样，渲染有效性是经验假设。

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
