# P99 全项目事实汇总 — Sol 执行 prompt

不计入20个方案。可在任意阶段执行，即便全部失败。必读 `research/shared_protocol.md` S4–S8、`research/execution_order.md`、`research/experiment_manifest.json`、progress与各run状态。无需重读文献全文。

## 目标

汇总P01–P20全部结果、失败、缺失、计算预算和公平性；按预设规则产生下一阶段队列。最终新颖性和研究价值交Astra，Sol不替代这一判断。

## 数据读取与一致性

只读 `runs/research/`、manifest/cache metadata。按run key去重，不按最新mtime随便取一个。检查每run能映射原模/输入/公式/代码版本；有变更则分表。状态NOT_RUN/FAILED/BLOCKED/COMPLETED/NEGATIVE均进入主表。失败不得当PSNR=0；缺字段记NA与原因。

通过逐帧CSV重算mean PSNR/SSIM/LPIPS/ΔMSE/teacher MSE/尾部/p95/末30帧/时间TDE，不信任手填summary；校验所有方法相机、时间、尺寸和GT hash相同，样本数量不齐禁止平均比较。PSNR是逐帧均值，不偷换成从全像素MSE转PSNR。种子重复确定性方法不当额外独立样本；帧高度相关，不用300帧当300独立样本算虚假p值。多场景报告逐场景及场景等权均值。

## 公平性必查

1. 按数量、实际bytes、活动Gaussian-frame、tile成本分别成组；总点/静动态点/实际serialized bytes、decode后GPU内存都列出。等K不等bytes，零值dense字段不算省内存。
2. 同输入采样与cache定义；fresh/stale、多轮重估、K2完整replay和降采样分别标记。新模型沿用旧cache如无固定proxy声明，标不合规。
3. FT及mask训练/局部恢复probe算总预算，P11的200+800与其它1000一致。无densify/隐含原模型参数恢复/额外points。失败重试也计时。
4. 独立V开发与test冻结；任何看test后改参数的run排除主结论，保留事实记录。官方checkpoint见过V的限制不消失。
5. 资源实测：统计/选择/FT/评测/解码/总时间，冷启动与缓存摊销，显存allocated/reserved，GPU总占用，render p50/p95与端到端。未测deployment不写加速。
6. 简单解释：P04对modular/P01/P02/P03；P08对additive/P07；P09对E与真实恢复；P10对max；P11对等步FT；P12跨budget；P13对留一删一；P14对幅度；P15对uniform；P16对FP16；P17对DCT；P18对无switch项；P19对same-cap；P20对uniform。缺必需对照只能列待补证。

## 输出

创建 `runs/research/aggregate/<timestamp_and_hash>/`：

- all_runs.csv（包括未运行和失败）、per_scene_budget.csv、pareto.csv、fairness_audit.csv、failure_catalog.md。
- budget_ledger.csv：每run/每缓存的GPU、CPU、wall时长、峰值、实际重试数及剩余额度；不能将多任务墙钟当GPU小时。
- report.md：先项目真实完成状态，再最有信息量正/负证据、简单解释、代理/实现/优化失败分类、代表性局限。
- next_queue.json：仅按S7/execution_order晋级规则排序，最多6/3候选，附各门槛实际值与未满足项。没有合格方案时空队列是合法结果。
- fixed_visuals/：按S1固定帧和新增误差最坏帧选择；统一色标/裁剪并带GT、reference、baseline、方法。时间主张附相同相机固定帧率视频；不得挑剪辑掩盖切换。
- astra_handoff.md：每个C候选一段“支持什么/不支持什么/最近先例差距/需要Astra判断”，绝不写“证明原创”或“达到论文标准”。

研究状态回填progress但不覆写历史run。汇总不启动GPU；缺四个资源匹配对照时，只在已授权批次及剩余4槽内交runner排队（GPU串行），否则明确待补。阶段2/3无授权不自动执行。工具/一般代码错误可修复；改变方法定义、预算或测试划分必须停止相关项并记录。

## 验收

20个ID在表中无缺项；所有COMPLETED能溯源；run数和小时可由ledger独立重算；缺失/失败/预算不公平不混入排名；再运行汇总产生相同数据表（仅路径timestamp可变）。最终答复仅给完成/失败计数、预算、报告和astra_handoff路径、少量真正待决策问题。
