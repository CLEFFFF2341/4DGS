# 共享实验协议 v1

这是 P00–P20/P99 的规范来源。相对路径均相对 `E:/4DGS`。本轮只准备文档；只有用户随后把 prompt 交给 Sol 并授权执行批次，Sol 才开始运行。A=必要基线/已有方法迁移，B=机制诊断，C=待证实研究候选；C 不等于已原创。

## S1 冻结输入与划分

主场景 S0=`cut_roasted_beef`，原始 M0=`pretrained/cut_roasted_beef/point_cloud/iteration_40000` 双 PLY+cfg_args，哈希见 project_facts。t=0..299，最终分辨率 1352×1014，黑背景、near=4/far=300、原 SH/cube/slerp/time padding。禁止只截前 30 帧当整个视频。

按原始 cam 文件夹名划分：压缩开发验证 V={cam01,cam02}；最终 test={cam00}；拟合/统计 C=其余实际存在相机（cam04 缺失等须按实际 inventory，不能补造）。V 是预训练看过、压缩阶段没用的视角，所有报告保留这个限定。P00 如两 V 不存在，选排序最小的两个非 cam00，并写 manifest 后冻结；后续不得自换。

统计视角 C4：C 排序的索引 floor(j*(len(C)-1)/3),j=0..3，去重。训练微调用全部 C。T24={floor(299*j/23),j=0..23}。机制评测 V×T24（48 图）；阶段2评测 V×T60，T60={floor(299*j/59)}（120图）；阶段3评测 V×0..299（600图）并最终 test×0..299（300图）。V 决定方法/超参，test 只在冻结后运行。统计用 338×254（长宽根据相机加载保持约 1/4，记录实际值），所有方法一致；评测始终 1352×1014，不能为了 OOM 偷降。

时间诊断固定窗口 W=[0..9],[70..79],[145..154],[220..229],[290..299]，V 两相机，共100图，仅对主张时间收益的方法额外跑，计时计预算。验证图固定 t={0,74,149,224,299}，另选按新增 MSE 排序最差三帧（并列按 cam,t），所有方法同规则。

S1=coffee_martini，S2=sear_steak（只扩展这两者；不得看测试成绩换场景）。阶段2先检查官方 Ex4DGS v0.1 同名模型、数据预处理、300帧、对应 cfg_args。如果尚缺，标 BLOCKED_DATA 并继续 S0；允许后续执行批次在明确授权下载/预处理时获取，禁止悄悄训练新模型替代。原始视频不能覆盖；派生帧放单独受版本标识的目录。跨场景效应不能由 S0 重复种子替代。

## S2 ID、模型与接口

初始 ID 为 `(source_checkpoint_sha256,kind,row)`，kind=static/dynamic。render concat 索引通过显式 map 映射 ID；每次物理删除后 map 重建。合并保留 parent ID 列表和新 UUID，类型转换保留 provenance；ID 不是当前 tensor row。

拟新增接口（未实现）：`load_reference(manifest)->ModelAdapter`；`collect(model,sample_manifest,spec)->CacheRef`；`select(method,model,cache,budget,seed)->SelectionResult`；`apply(selection)->new_model`；`save_bundle/load_bundle`；`evaluate(bundle,split)->metrics`。ModelAdapter 给出静/动态数量、稳定 ID、所有 per-point 张量清单、时间属性。SelectionResult 含 kept IDs、类型/轨迹/属性变换、实际成本、父模型 hash。不同方案独立 `research_impl/methods/pXX.py` 和 `configs/research/pXX.json`，共享 bug 修复后统一回归；禁止覆盖另一方案。

接口测试：全保留渲染 max abs≤1e-6（若原生 CUDA 非确定性底噪超过，先测3次底噪，以5倍底噪作阈值并固定记录；不随方法放宽）；随机删静态/动态各10个与手工 tensor gather 一致；检查 True=删除；所有属性、缓存列、optimizer 状态长度一致；空 static/dynamic 独立处理；save-load 后 ID、tensor、时间 t=0,0.5,149,299 与输出一致；重跑选择 IDs 位等；GPU atomic 统计允许绝对1e-7+相对1e-5但近并列需稳定 CPU FP64 汇总/量化排序。NaN 是失败，不当零。

纯推理剪枝不要为 optimizer 随便分配全量 Adam；审计后编写无 optimizer 的 gather adapter。微调时重建 optimizer，所有对照同样重建，不声称沿用官方状态。

## S3 统一统计（不要混淆代理）

记 N 点，L=96 个 C4×T24 视时样本；r=(v,t,pixel)。alpha a_ir 是 rasterizer 实际 pixel alpha，T_ir 是该点前透射率，w_ir=T_ir*a_ir∈[0,1]，颜色 c_ir∈R³。无贡献为0。K0 基础张量包含时间位置、旋转、scale、opacity、SH、投影范围；K1 包含 s_it=mean_v(mean_pixel w_ir)∈R^(N×24)、m_i=max_r w_ir、hit counts、tile touches、各时可见性、体积；K2 包含单删除向量 d_ir=I_full(r)-I_without_i(r)=w_ir*(c_ir-b_after_i,r)，e_it=mean_(v,p,RGB) d_ir²。K2 定义是相对 teacher 的删除失真，不是对 GT 的损失增量。

P00 统计实现按能力分层：先 CPU double 的有序 alpha compositing oracle；再采样 ray 的 CUDA 列表/统计扩展。只用 dominant ID 不能产出 K1/K2。采用相同投影/过滤，保存每条采样 ray 的所有 alpha≥1/255、投影深度顺序贡献者（包括原 renderer 早停后的候选以验证反事实）；alpha cap 与 covariance correction 不变；排序 tie 用原规则。仅用于统计的无早停重放需与原 renderer 对齐，并记录早停误差。不能从 radii 推测 w。

K3 稀疏覆盖 A∈R_+^(U×N)：统计图按8×8 patch，每 patch 固定4×4等距 pixel（坐标中心、floor、clip），u=(v,t,patch)，A_ui 为16 rays 的 w 均值；空边界按实际采样数，保留非零 CSR（不先 top-k 截断）。对应 d 的样本/patch summary 可从相同 replay 得到。每个 view/time 权重相等，patch 权重为实际面积/图像面积。存储 >8GB 时先 chunk/mmap；仍不行启用预定义 2×2 rays 并给所有比较方法重建同级缓存，标 low_sample，不与16-ray正式结果混表。不能构造 N×U dense。

K4=轨迹邻接/拟合统计，K5=小规模干预重渲染结果，K6=梯度/优化过程统计。K5/K6 不得在模型变动后冒充静态缓存。误差模型 replay 残差>1e-4 RGB 或单删真实重渲染相对误差>5%（对真实d²<1e-10使用绝对1e-8）时，K2依赖方案先修复；仍不行标 PROXY_INVALID，不替换成 w²。

缓存 key=SHA256(双PLY+cfg、代码 commit+dirty diff hash、CUDA扩展hash、统计公式版本、相机/图片列表及内容hash、所有时间、实际尺寸、背景/裁剪/SH/interp、采样seed/dtype/filter规则)。共享需要以上完全相同。变动后重新判断字段依赖：只从同一原始 w 计算 mean/max 无需重渲染；prune/merge/FT/quantize/新的稀疏截断/公式引入新量必须新 key。参考原模 K1 可作为固定选择输入，但不能称为剪后真实贡献。每缓存记录构建时间/字节/峰值；同时报告冷启动和 amortized 成本。

阶段0进行实测抽样误差：2帧×1相机，比较16 vs64 rays/patch、原生统计尺寸 vs二倍尺寸；报告排序 Spearman、topK Jaccard、单删误差、近阈值点换序。Spearman<0.8 或 Jaccard<0.7 时 flag；最多一次统一升级（64 rays 或二倍尺寸中针对原因选一个，全方法同步）。超过预算则依赖方案 BLOCKED_CACHE，不宣称假设被否定。

## S4 数量、表示和资源预算

数量赛道：b=0.5，K=floor(b*N)=124447；阶段2 b={0.25,0.5}，阶段3同。除 P12，按原比例给 static/dynamic 配额：Ks=floor(K*Ns/N),Kd=K-Ks，遇组容量上限向另一组移交。分数高的保留，tie 按 stable ID；零分补足到K。随机种子0；阶段2种子{0,1}；阶段3{0,1,2}。只对真正随机过程改变种子；确定性重跑记验证，不伪造独立样本。

方法明示取消kind配额（P19）或改变kind（P14）时依其定义，分表审计。stochastic greedy候选样本大小统一取min(剩余候选数,ceil((N_group/K_group)*log(1/epsilon)))；K_group=0直接跳过，不除零。邻域不足k取全部实际邻居。世界空间尺度L统一为原模型t149全部点bbox对角线，L≤1e-12设1；用哪个集合/时间算尺度写进cache key，不随删点更新。P05特征标准化前的位置归一化也用此L。

表示赛道 P14–P17/P20 默认保留全部点。只压部分属性时，不能要求整个模型减半：定义 B_variable 为被本方法处理的原payload，B_fixed=B0-B_variable，阶段1目标为 B_fixed+0.5B_variable 的实际可加载bundle（含索引、padding、headers、codebook、解码必要文件，排除评测日志）。P14变量部分是全dynamic比转换成static多出的payload；P15为动态position+rotation；P16/P17为动态position；P20为全部SH AC。预算按方法定义冻结，因此不同方法总bytes不同，不能仅比较PSNR称公平。P16编码固定，允许报告其实际接近目标的字节点；阶段2两预算使用其可实现的固定编码模式/零FT与统一FT分表，若无第二真实率点不造结果。

与P01/P02实际字节≤相同上限的数量模型比较：按E或贡献分数排序，二分保留前缀数量，只检查实际bytes、不在V搜最优K；最多8次字节探测，无渲染不算正式run。第一次资源匹配评测占execution_order中4个共享额外槽，超出则待补证。如无法达到目标，报告实际RD点/目标未达，不叠加未规定压缩器。P13合并仍是数量赛道；P18是活动点时间预算赛道；P19分别字节和tile-cost预算，不能强制同时等K。各赛道单列，不按PSNR总排名。

所有运行报告：唯一点数、静/动态点数、每帧实际活动点数、已加载tensor字节、峰值显存、序列化原始/实际打包字节。外部 ZIP 与内部部署文件分列。零值仍在dense tensor不算省显存；解码为原dense模型只算省磁盘；少点但大覆盖不必更快。

## S5 评测与时间指标

用已验证仓库 PSNR/SSIM 实现和 Alex LPIPS v0.1，RGB [0,1]（LPIPS adapter 只做一次[-1,1]映射，核验内部normalize）；alpha/background相同。逐帧记录 PSNR、SSIM、LPIPS、GT MSE；新增误差 ΔMSE_t=MSE(I_method,GT)-MSE(I_ref,GT)，teacher D_t=MSE(I_method,I_ref)；二者不同，ΔMSE可负。同时存 PSNR drop、最坏10%帧平均drop、p95 ΔMSE、最差帧、末30帧均值。无穷PSNR仅零MSE数值控制中允许，汇总显式处理。

时间指标：固定相机，残差 E_t=I_method,t-GT_t；TDE=mean|E_(t+1)-E_t|；增量 TDE 与 reference 相减。再报告 teacher residual Q_t=I_method,t-I_ref,t 的 mean|Q_(t+1)-Q_t|。它们是固定视角误差变化指标，不是运动补偿感知闪烁真值。使用固定窗口W，只对相邻帧配对。快速运动可能提高这些量，必须附GT帧差和静/动态ROI分组；ROI由相邻GT的逐像素RGB平均绝对差超过固定0.03产生，3×3膨胀，并标为运动近似非语义真值。若声称“减少闪烁”，阶段3必须提供全300帧固定视角视频并做配对检查；没有可信光流不捏造 warped metric，也不把已有render opticalflow零占位当GT flow。

延迟测量：单GPU锁、无其他GPU任务；batch=1，预加载camera，warmup20，100次循环固定10个(相机,t)输入；CUDA events+同步计 GPU core，wall clock计含模型时间属性生成的端到端 render；p50/p95/均值，IO另计，报告GPU/driver/torch/扩展、显存allocated/reserved峰值与设备总占用。3次测量块，中间重置峰值，时间不包含LPIPS。预训练日志27FPS不当新方法基线。统计/选择/解码/微调/评测/总耗时都单列。

## S6 微调与搜索公平性

阶段1、2默认无微调；P11有规定的掩码优化，P09有仅诊断的局部求解，它们所有优化成本必须记入。P11必须有相同步数普通FT对照，不与零FT混称优势。

阶段3共同FT=1000步、batch1、按seed对C×全部t采样、0.8 L1+0.2(1-SSIM)，Adam，原cfg标量LR；位置/动态位置用官方scheduler在40000时的值并固定，其他按cfg固定。不 densify、不reset、不改时间长度、不自动prune、不改变K。P11选择阶段200步+800步FT，总1000；其他1000步FT。P07多轮选择不插入FT，故仍1000。可变表示不能按同样优化参数更新的模型，只提供冻结结构可训练参数adapter；若不能，单列零FT，不宣称赢FT赛道。所有方法先报零FT，再报FT。

每份prompt列完整参数行，行数是上限，不做笛卡尔积。训练C用于统计/拟合，V用于选行；阶段1选一个主行（有分组机制对照则保留指定对照），阶段2冻结到新场景。不能看到test后改参数。容许每方案1次实现修复重跑（仅原矩阵）；不容许无上限调试来“调通收益”。

## S7 分阶段决策与成本

run定义=一个scene/method/参数行/budget/seed/FT状态生成模型并完成规定评测；缓存构建和干预probe另计，不能藏在run数之外。

| 阶段 | 上限 | 晋级与停止 |
|---|---|---|
| 0 | P00含首次共享cache合计最多4 GPU小时；每方法smoke最多10 GPU分钟，总20×10min；CPU oracle/读盘另记 | 全保留/索引/序列化/统计oracle必须过；2帧 smoke 测峰值；不通过是实现问题。 |
| 1 | 表中51个正式run上限；单run≤45 GPU分钟（含本方法额外统计/优化/评测），选择CPU≤45分钟；共享cache≤4 GPU小时且仅建一次。合计≤46 GPU小时含阶段0/smoke/共享cache，实际预期更低但未经实测 | 至少给每个可行机制一次试验，不以第一失败淘汰。对照公平后，PSNR≥对照+0.10dB且尾部不劣0.2dB、LPIPS不劣0.002；或PSNR在-0.10dB内且目标资源改善≥10%；时间机制另要求TDE减少≥10%且质量门槛通过。都是预算决策阈值，不是显著性。 |
| 2 | 最多6个晋级方案+P01最佳简单行/P02共≤8方法，3场景×2预算×2seed=≤96run，每run≤30 GPU分钟；共享cache每新增场景≤3小时；合计≤54 GPU小时 | 至少2可用场景方向一致且无>0.5dB尾部灾难，公平性通过；选≤3候选。若只有1场景，标“探索性、泛化未验证”，不假过跨场景门槛。 |
| 3 | ≤3候选+3强对照（P01最佳、P02、P11；重叠去重）×3场景×2预算×3seed=≤108 run，每run含FT≤60 GPU分钟；合计≤108 GPU小时 | 冻结代码/参数/排序规则后一次test；不得边看test边决定谁参赛。原论文完整pipeline另需预算，不挪用本阶段。 |

正式run上限=51+96+108=255；额外共享未剪baseline评测最多9次（3阶段×3场景），资源对照最多4次已含阶段1表；总评测run账本上限264。这9次baseline和所有probe仍在各阶段GPU小时上限内，并不额外加时。阶段2预算：数量={0.25N,0.5N}，表示字节={B_fixed+0.25B_variable,B_fixed+0.5B_variable}，活动/tile={0.5,0.75}参考成本，P19byte={.25B0,.5B0}；不交叉乘积，阶段3相同。没有可实现第二率点的方法只跑一个，减少实际run。数量/字节分组分别筛选，≤6/≤3是合计名额，可行的不同机制优先于同一家族小优势。原基线缓存可复用但每个原模型新hash必须重建。

另预留24次失败修复run（每方法1次+P00四次），每次≤30 GPU分钟，共12小时；失败也计总资源。全项目硬上限220 GPU小时（阶段上限和余量在此以内），到顶保存状态交P99。CPU选择全项目≤40小时、单方法实现/调试≤6 wall小时（P00≤12小时）；耗尽则BLOCKED_IMPLEMENTATION，不占GPU死等。额外下载/预处理最多8 CPU小时/场景，不含网络等待；缺官方checkpoint则停止该场景。本轮不执行这些预算，它们是未来统一批次的建议上限。

显存门槛22GiB allocated或设备总占用23GiB：先释放暂存，CPU缓存/分块、batch1、chunk大小减半，最多2次；仍超限停止该方法本规模，不能降评测分辨率/减少保留数。资源超时先做2帧+1%点的数值探针（非质量结果），记录 bottleneck；不能把小规模结果当正式场景结果。

失败分层：IMPLEMENTATION_ERROR（索引/梯度/IO）→ PROXY_INVALID（数值正确但低相关）→ OPTIMIZATION_FAILED（目标未收敛、soft/hard差大）→ NEGATIVE（正确/公平且机制证据不支持）。必须保留已完成诊断后继续无依赖任务；负结果需至少一个同矩阵诊断/修复机会，不默认加超参。

## S8 输出与恢复

每run=`runs/research/<Pxx>/<scene>/<budget>/<seed>/<config_hash>/`。必须有 resolved_config.json、provenance.json（git+diff+model+cache+数据hash）、commands.txt、stdout/stderr、status.json（NOT_RUN/RUNNING/FAILED/BLOCKED_*/COMPLETED/NEGATIVE）、timing.json、resources.json、per_frame.csv、summary.json、kept_ids/变换映射、可重载模型、固定可视化和method_report.md。原数据/原checkpoint/既有runs只读；写结果先临时目录后原子rename；不可覆盖已完成run。

progress.json 只记录队列、依赖、完成hash、GPU预算累计、失败和重试次数；GPU lock记录PID与启动时间，恢复核验PID实际存活再清陈旧锁，不杀不明进程。新代码导致cache不兼容时停依赖任务并更新版本。P99读取全部，包括未跑/失败项，不仅赢家。
