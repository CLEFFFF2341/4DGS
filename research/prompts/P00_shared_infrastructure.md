# P00 共享基础设施 — 首先交给 Sol

你负责实现、调试、运行并如实记录；Astra的方案已经写入研究文档。只在用户当前授权批次内执行，不假设研究文档等于已授权220小时全计划。P00不计入20个方案。

## 1. 目标与必读范围

工作目录 `E:/4DGS`。读 `research/project_facts.md`、`research/shared_protocol.md` S1–S8、`research/execution_order.md`。只需读 Ex4DGS 本地PDF pp4–6和GaussianPOP pp3–4（准确文件名见literature_matrix）。不要重读29篇论文；不自行选唯一研究方向。

目标是建立不可变参考模型、分割与统一测量，使P01–P20独立可执行。用户有数学/PyTorch基础，对Ex4DGS实现不熟；报告用清楚的字段/真实命令说明接口，不留下隐含知识依赖。

## 2. 先确认项目事实

核对HEAD和dirty diff（保留已有修改，不reset）。校验双PLY哈希/数量，cfg_args只按安全parser读取Namespace参数，不eval不明代码。确认当前环境、CUDA扩展与PyTorch是否可用；不要重装能用的环境。GPU实验前核对没有另一个任务占卡。原始文件只读，输出全部独立目录。

真实源码定位：

- `scene/__init__.py:Scene,getmodel,getTrainCameras,getTestCameras,set_sampling_len`。
- `scene/dataset_readers.py:readN3VInfo` 的cam00分割、路径与timestamp。
- `scene/c_gaussian_model.py` 属性清单、`get_*_at_t`、`save_ply/load_ply`、`prune_points/_prune_optimizer`、`training_setup/update_learning_rate`。
- `gaussian_renderer/__init__.py:render` 的concat、near/far/background和dominant_idx返回。
- `submodules/diff_gaussian_rasterization_df/{diff_gaussian_rasterization_df/__init__.py,cuda_rasterizer/forward.cu}`及相关bindings。缓存统计API当前不存在。
- `render.py`、`utils/image_utils.py`、`utils/loss_utils.py`、`lpipsPyTorch/`。

将实际tensor字段、shape、kind、是否per-keyframe、保存字段对应写 `runs/research/P00/interface_audit.md`。若未确认函数签名，先读源码再实现，勿按文档猜。

## 3. 新建公共接口与manifest

拟新增 `research_impl/{adapter.py,registry.py,cache.py,evaluate.py,finetune.py,runner.py}`（命名是建议，可调整一次后在interface_audit中精确映射）；`configs/research/`每方法独立配置。adapter遵循shared S2，immutable加载/gather/变换/保存；inference gather不强制创建Adam；所有per-point张量与buffer明确注册，未知字段必须报错不能悄漏。

冻结 `research/manifests/reference.json`、`splits.json`、`samples.json`、`environment.json`：双模型hash、原cfg与路径override、全部真实相机/图片路径/hash、cam01/cam02压缩验证、cam00最终测试、C/C4、T24/T60/W、尺寸、采样坐标、软件版本与CUDA扩展hash。时间与数量以实际PLY/cfg核实。参考+run代码快照均能重建。

建立GPU lock、progress.json和预算ledger，支持按execution_order自动继续、失败跳过、重启检查。run key由配置/输入/代码决定。测试集入口必须检查 `frozen_test_manifest`，未冻结则拒绝test请求；P00不能再次批量跑cam00调指标。历史cam00复现只作为已知记录，不反馈方法参数。

## 4. 分层统计缓存与oracle

先做K0和CPU double alpha-composite oracle；再K1/K2/K3。精确定义/采样量按shared S3，不将radii或dominent_idxs冒充完整贡献。必要时为采样rays新增只读CUDA统计路径，默认render输出保持不变；统计关闭时不能改变原训练/渲染。

CPU oracle至少覆盖：0点、1点、同色前后点、遮挡后显露、透明点、clip alpha=.99、背景非黑、颜色误差正负抵消。比较ray循环、解析单删与暴力重渲染这三种独立计算。

最小GPU统计：C4第一个相机、t=0/149两帧。先对32个真实ID核对单删K2；同时含低/高贡献与静/动态，按E分位或ID固定取，不手挑好看点。原renderer早停后深层点可能因删点重新显露，需完整candidate ray replay并比较原forward；误差阈值按S3，失败不给K2挂合格标记。

K1可独立完成：s_it/m_i/opacity/volume/tile碰撞；K2需suffix背景重建；K3需CSR与每ray/projection支持，不输出dense U×N。CPU RAM峰值≤可用RAM70%，缓存上限8GB/模型级；超限chunk/mmap，再按S3预定义低sample统一降级。如果无法实现，写capabilities.json分别标available/blocked，使轨迹编码方案仍可执行。

每个缓存写schema、输入/公式/renderer hash、sample manifest、shape/dtype、checksum、耗时、CPU/GPU峰值。只有同key复用；cache文件一半写入不能被runner当有效。

## 5. 统一评测与固定预算微调入口

P00创建reference在V×T24的指标与延迟（最多1正式未剪baseline槽），PSNR/SSIM/LPIPS输入范围按S5。LPIPS权重缺失时写明确阻塞，不填0；不能用其它backbone替换。支持逐帧CSV、ΔMSE、teacher MSE、时间窗口W和尾部指标。原图/渲染均用同resolution/crop/bg。

测GPU core和包含time-interp/gather/decode必要计算的wall latency；统计/选择/FT/评测各阶段计时，baseline没有“免费统计”假设。保持未剪checkpoint不变。

建立S6 FT入口，固定checkpoint全部duration，禁用密度控制/reset/type conversion；gradient step单卡batch1。P00只允许最多10步FT smoke，保存到P00临时run验证梯度/数量/可重载，不把这份训练后的模型当参考。需1000步FT的正式实验留阶段3。

## 6. 正确性门槛

shared S2全部通过才正式选择：全保留、混合kind删除、mask方向、所有属性长度、optimizer重建、空kind、save-load、stable IDs、确定性、连续/边界时间。对score采样做16/64ray与双尺寸diagnostics，不通过按S3修复或cache block。

真实render的`compute_cov3D_python/convert_SHs_python`分支存在可疑调用参数，不为省事切开未经验证分支。默认用cfg里的False。不借修复此任务重构所有训练算法。

## 7. 最小运行预算与验收

CPU数值测试→2帧原模smoke→2帧统计/32ID单删→V×T24 reference评测→10步FT smoke→全T24共享cache。P00与其首次共享cache合计≤4 GPU小时（不是各4小时），实现/调试≤12 wall小时；单阶段显存22GiB/设备23GiB门槛，chunk最多两次再停，质量评测分辨率不降。GPU时间未知，先给实测估计和预期完整cache时间；预计超限先完成K0/K1和独立核心，不无限编译/计算。

通过验收产物：可重载reference/子集bundle（保存派生副本）、全部数值检查、capability矩阵、冻结manifest、参考metrics、缓存key、执行CLI帮助、能够run one method的registry样例、队列状态和预算ledger。未完成K2/K3不假装完成整P00，应报告CORE_READY/PARTIAL_CACHE，自动执行不依赖该cache的任务。

## 8. 连续执行与输出

输出路径 `runs/research/P00/`，S8完整记录；`research/progress.json`维护恢复。若用户授权P00+阶段1，按execution_order继续P01–P20，失败留诊断后跳过依赖链，最终自动P99。若仅P00，到此结束并交付真实命令/能力矩阵，用户不用再问实现细节。

关键数据缺失、定义冲突或必须改变假设时记录BLOCKED，不为通过而改验证标准。不要改原数据/checkpoint、既有复现结果或其它方案输出，不向外发消息，不启动并行GPU，不宣称当前任一研究候选已有效。
