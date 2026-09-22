# 项目事实与边界

核验日期：2026-09-22。此次工作只读取代码/结果、检索公开来源、提取论文、编写文档；未安装环境、修改算法或运行 GPU 实验。研究文档不是实验结果。

## 已核验

| 项目 | 事实及证据 |
|---|---|
| 仓库 | `E:/4DGS`，origin `https://github.com/juno181/Ex4DGS.git`；HEAD `1e4539eea2d2a15d286db003e0c4ae8cca4ec792`。目录名 4DGS 不代表 Wu 等人的 HexPlane 4D-GS。 |
| 源码状态 | 开始时 `git status --short` 只有未跟踪的论文、数据、模型、工具和结果；无 tracked 算法修改。未发现 AGENTS.md。 |
| 原始模型 | `pretrained/cut_roasted_beef/point_cloud/iteration_40000/{point_cloud.ply,dynamic_point_cloud.ply}`，旁边的 `cfg_args` 是配置真源；不能用基础训练 JSON 覆盖它。 |
| 静态 PLY | 185,033 点，65 个 float 属性，48,110,189 bytes；SHA256 `d0b66a2d2ae2dca1d864335ae015cf0f9ca212a04b8c6ebb2b11e23ab3753171`。 |
| 动态 PLY | 63,862 点，315 个 float 属性，80,475,920 bytes；SHA256 `3d48eb4eb4e65858fb5e88b2d8be03942ab29c2b242aa3bda223170e7f209e98`。 |
| 合计 | N=248,895；两 PLY 共 128,586,109 bytes。属性含冗余/序列化字段，不等于 GPU trainable 参数量。完整头信息见 `checkpoint_inventory.json`。 |
| cfg_args | model=cubic，interp_type=cube，rot_interp_type=slerp，duration=300，time_interval=10，time_pad=2，start_timestamp=0，resolution=2，SH degree=3，near=4，far=300，black background。保存路径含 Linux 旧路径，运行时仅重定向路径。 |
| 已复现 | `runs/cut_roasted_beef/reproduction_render_metrics.json`：300 帧，t=0..299，1352×1014，PSNR 33.73134496，SSIM 0.95764688，平均 core render 0.03703676 s。`reproduction_lpips_alex.json`：Alex v0.1 0.04043770，与原作者文件差约 3.37e-6。这里只复述已有日志，未重跑。 |
| 数据 | cut_roasted_beef 已有 PNG、poses、colmap_0。另五场景 coffee_martini/cook_spinach/flame_salmon_1/flame_steak/sear_steak 目前主要为 MP4 与 poses，不能视为全部预处理完成。只找到 beef 的预训练 PLY。 |
| 历史记录 | `EXPERIMENT_LOG.md` 记录下载、校验、环境与复现过程；第 38–41 节记录其余数据包完成。原始数据在 scene 层不完整等价于不可运行，不从“下载完成”推导“可训练”。 |
| 环境 | 日志记载 `C:/Users/CLEFFFF/anaconda3/envs/Ex4DGS/python.exe`。本轮未验证 CUDA/扩展可用性，也未测当前显存。单卡 4090 24GB 是用户研究条件。 |

已有渲染不是剪枝成果；尚无本计划稳定 ID、贡献缓存、统一剪枝实验、微调、阶段筛选或统计显著性证据。`runs/` 的复现副本不能替代原始 checkpoint 身份。

## 已核验接口（给 Sol 的地图）

- `scene/__init__.py:Scene`、`getmodel`、`Scene.getTrainCameras`、`Scene.getTestCameras`、`Scene.set_sampling_len`：相机和时间加载。默认可 shuffle，研究入口必须显式关闭并按路径/timestamp 排序。
- `scene/dataset_readers.py:readN3VInfo`（约 520 行后）：通过路径包含 `cam00` 划 test，其余为 train；不能把日志中的整数 camera ID=1 误解为 cam01。
- `scene/c_gaussian_model.py:get_xyz_at_t`（170）、`get_static_xyz_at_t`（178）、`get_dynamic_xyz_at_t`（182）：默认拼接顺序 static 然后 dynamic。“静态”位置实际为 `_xyz + _xyz_disp*t/duration`，不一定完全静止。
- 同文件 `get_rotation_at_t`、`get_scaling`、`get_features`、`get_opacity_at_t`：按 t 得到真正参与渲染的属性。动态 opacity 有时域 envelope；仅 base opacity 不代表可见性。
- 同文件 `save_ply`（513）、`load_ply`（560）：双文件配对。`prune_points(static_mask,dynamic_mask)`（715）中 True 表示删除，内部取反；依赖 optimizer，纯推理加载后不能盲调。`_prune_optimizer` 同步 Adam 状态；所有缓冲区仍需审计。
- `gaussian_renderer/__init__.py:render`：返回 render/depth/acc/`dominent_idxs`/radii/visibility_filter 等。`radii>0` 只是投影未剔除，不是遮挡后可见；`dominent_idxs` 只存单个最大 alpha*T ID，不是全贡献矩阵。
- `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`：renderCUDA 中 alpha cap=0.99，alpha<1/255 跳过，test_T<1e-4 时早停；约 411 行写最大贡献 ID。精确统计必须尊重过滤、深度排序、cov.w 与背景。全列表重放需说明早停差异。
- `render.py:render_set,render_sets`、`utils/image_utils.py`、`utils/loss_utils.py`、`lpipsPyTorch/`：评测复用来源。新入口不能默认执行整个官方 test。
- `train.py`、`scene/c_gaussian_model.py:training_setup,update_learning_rate`：微调需禁用 densify、自动 prune、opacity reset、静动态转换、progressive growing。官方 train.py 不能未经审计直接当固定预算微调器。

## 未核验 / 需 P00 固化

所有 `research_impl/`、方法注册、cache API、配置 schema 都是本次设计的新接口，当前不存在。不得在报告里称其已实现。需检查空动态集保存、checkpoint 是否有 optimizer、CUDA build ABI、LPIPS 输入范围与权重文件、相机文件完整性、半分辨率加载细节、时间 padding 数量、RAM/磁盘峰值。不需要为普通细节回问 Astra。

## 数据泄漏限制

当前官方 checkpoint 已用非 cam00 相机训练。新划 cam01/cam02 只能作为“压缩开发验证”，不能称为预训练未见视角。cam00 已有复现指标，不能假称研究者完全没见过；从现在起封存其逐帧结果用于方法选择，阶段 3 规则冻结后一次性最终评测。真正独立的泛化结论需要额外场景或重新训练留出验证相机的基模；本批不默认从头训练。

## 目录约定

文献：原始 `Papers/` 只读；补充公开 PDF 在 `research/sources/`；逐页文本在 `research/paper_text/`。科研实现将来放 `research_impl/`，配置 `configs/research/`，实验结果 `runs/research/`，缓存 `research_cache/`，调度状态 `research/progress.json`。本轮最后四处仅设计，未执行创建实验。
