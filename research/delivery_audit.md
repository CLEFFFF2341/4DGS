# 文档交付审计

2026-09-22，状态为研究设计交付，不是执行结果。

- 必需5份根文档与P00、P01–P20、P99均存在；20方法prompt均含1–9节。
- 自动核验论文文件名和experiment_index的prompt链接，无缺失引用。
- 阶段1方法行47+共享resource-match槽4=51；后续96/108与未剪baseline9共264正式评测run上限，另24修复槽。实际跑数会因复用/阻塞/淘汰减少。所有probe、cache、失败耗时单列并受预算限制。
- 29份本地论文完成摘要筛查，最相关方法段按literature_matrix的页码细读；无L3全文精读声明。下载4DGC和Lazier Than Lazy Greedy成功；CDGS PDF下载失败（IncompleteRead），用官方arXiv HTML核验并引用。没有将失败PDF列作已读本地材料。
- 原论文/新增PDF共31份SHA256见source_checksums.json；原模哈希、点数及PLY头见checkpoint_inventory.json。OMG4 p6、GaussianPOP p3已渲染核对关键公式，qa图仅作阅读证据。
- 明确修订：表示压缩预算按可压字段，避免只压位置却要求整模型减半；K/bytes/active-frame/tile四赛道分离；同等kind配额与取消配额的实验分开；stochastic candidate不足时取min；P08限制64边界候选/组和16swap以约束机制实验成本。
- 当前官方基模见过开发验证相机，不称严格未见泛化；cam00历史复现已见，从本计划起禁止调参/选择依赖该test。
- 最后git diff --exit-code通过：未修改tracked算法。只新增research文献/文本/文档/生成辅助脚本；没有GPU实验、没有安装环境，没有修改原数据/checkpoint/既有结果。

局限：候选未获实验支持；未核验每篇论文全代码/附录；P08/P11统计或CUDA实现仍可能超时，不能保证全部20方案都能在24GB与时间额度完成。外部原论文完整pipeline复现尚未执行，不能据模块迁移宣称全面SOTA。上述局限在P99强制保留。
