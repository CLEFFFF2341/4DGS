# P00 状态报告

状态：`CORE_READY / PARTIAL_CACHE`。记录日期：2026-09-23。最终测试相机 `cam00` 仍封存，未用于本轮方法选择。

## 已完成

- 安全解析 `cfg_args`，重定向本地数据路径，不执行配置文本。
- 双 PLY 哈希和数量复核：static 185,033，dynamic 63,862，总数 248,895。
- stable ID、无 optimizer 子集 gather、双 PLY bundle、重载校验与方法 registry。
- 8 类 FP64 alpha-composite / 单删除 oracle；解析结果与逐点暴力删除最大误差不超过 `5.56e-17`。
- 全保留渲染 max/mean abs 均为 0；固定 seed 混合删除 10+10 个点的 15 个逐点 tensor 全部与手工 gather 完全一致。
- 原始 bundle 在 t={0,0.5,149,299} 保存重载渲染 max abs 均为 0；空 static 与空 dynamic 两种模型均可渲染。
- 真实 GPU smoke：cam03 的 t=0、149，1352×1014；峰值 allocated 935,922,688 bytes。
- V={cam01,cam02}×T24 reference：48 图，PSNR 35.65164248，SSIM 0.96526255，LPIPS-Alex 0.04315178，GT MSE 0.0002800102。
- 延迟：20 warmup 后 3×100 次；三块 wall mean 为 11.355、11.402、11.381 ms，峰值 allocated 999,419,392 bytes。
- K0 T24 缓存：key `a168c2921438bfad56a8d7f5b2a1da673dcbbb988e70aad45fc7afe76c38adde`，243,919,918 bytes。
- 固定点数 10 步 FT smoke：点数未变，bundle 可重载且渲染 max abs=0；峰值 allocated 2,311,789,568 bytes。

## 新发现的数据事实

非 cam00 目录当前只含 `000000.png`，完整 300 帧保留在同名官方 MP4。评测器不向原数据目录补写 PNG，而是复制固定相机位姿并按真实时间戳顺序解码只读 MP4。cam01 的 MP4 第 0 帧与现有 PNG 逐像素完全一致。顺序解码仍出现过一次 ffmpeg/H.264 宏块警告；所有请求帧都返回成功，因此暂记数据质量警告，后续哈希与扫描继续核验。

## 尚未完成

- K1/K2/K3 需要 rasterizer 输出真实 `T*alpha` 贡献、完整候选 ray replay 与 CSR patch 覆盖；当前 `dominent_idxs`/radii 不足，未用代理冒充。
- 16/64 rays 与双尺寸相关性诊断依赖上述统计扩展。
- 10 步 FT 已验证训练入口，但正式 1000 步只属于后续授权阶段。

主要产物位于 `runs/research/P00/`，缓存位于 `research_cache/cut_roasted_beef/`；二者按约定不提交 Git。关键实现、配置、manifest 和本报告提交到实验分支。
