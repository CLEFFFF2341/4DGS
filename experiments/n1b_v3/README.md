# N1b-v3：共享时间轨迹结构验证

本目录保存 2026-10-26 的持续研究。最终报告见 [RESULTS.md](RESULTS.md)，
逐阶段的成功与失败见 [RESEARCH_LOG.md](RESEARCH_LOG.md)。

有效候选是八个时间节点共享的仿射中心校正，增加96个参数；它改变时间表示，
曝光积分仍用普通Gauss。没有把原N1b解析控制变量或新分区采样称为成功。

## 主要文件

- `models.py`：原生模型、时间中心解耦、时间扭曲、曲率、共享分段中心校正及消融。
- `search.py`：公平训练、保存完整模型、统一Gauss32评估；支持固定更新或实测时间预算。
- `support_schedule.py` / `schedule_audit.py`：预算内分区及其负面重建结果的机制验证。
- `gradient_audit.py`：真实训练损失的原生分支梯度收敛检查。
- `prepare.py`：只使用训练双目观测建立共同初始化。
- `confirm.py`：冻结协议的九组独立确认与两种预算；源码哈希不符则拒绝执行。
- `audit.py` / `assess.py`：独立LPIPS、积分收敛、模型哈希及预设成功判据。
- `confirmation01_frozen.json`：确认前保存的源码与协议哈希。
- `results/`：机制检查、独立审计、判据及固定视图对照。

大模型与逐运行记录在 `E:\4D_Gaussian\output\n1b_v3\`。新运行目录必须不存在，
所有旧结果及生产源码保持原样。`confirm.py` 的路径固定用于本轮审计；下列入口
可在新目录复现候选与对照，而不覆盖已完成的确认。

## 在新目录复现

在仓库根目录执行：

```powershell
$env:PATH='E:\4D_Gaussian\.conda\nerficg\Library\bin;'+$env:PATH
$env:TORCH_HOME='E:\4D_Gaussian\.cache\torch'

# 原始DSEC帧和标定已经下载；新种子与新输出目录。
& .conda/nerficg/python.exe -m experiments.n1b_v3.prepare --root dataset/dsec_dynamic/zurich_city_05_a --start 120 --seed 6 --out output/n1b_v3/my_initial

# 同优化器、损失、6400次更新/渲染；两臂最终模型统一Gauss32评估。
& .conda/nerficg/python.exe -m experiments.n1b_v3.search --run output/n1b_v3/my_initial --out output/n1b_v3/my_comparison --optimizer core --steps 6400 --arms native:1:0 affine_spline:1:0.00003

# 独立感知指标与积分收敛。
& .conda/nerficg/python.exe -m experiments.n1b_v3.audit --runs output/n1b_v3/my_comparison --output experiments/n1b_v3/results/my_audit.json

& .conda/nerficg/python.exe -m unittest experiments.n1b_v3.test_models -v
```

适用范围为真实DSEC短片段的固定容量有限曝光帧重建。没有自动接入生产Trainer、
增密/剪枝、百万高斯、真实瞬时去模糊或未知标定流程；也不主张论文新颖性。
