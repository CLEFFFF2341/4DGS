"""Research document authoring helper. Does not import torch or run experiments."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
items=[]
def add(i, slug, name, kind, hypothesis, related, refs, method, checks, matrix, evidence, risk, cache, cost, order, n):
    items.append(dict(i=i,slug=slug,name=name,kind=kind,hypothesis=hypothesis,related=related,refs=refs,method=method,checks=checks,matrix=matrix,evidence=evidence,risk=risk,cache=cache,cost=cost,order=order,n=n))

add(1,'score_baselines','可复用标量与采样基线','A',
'若复杂方法只是偏好高贡献或改变采样分布，简单打分应解释大部分收益；观察各规则的ID重叠、平均和尾部误差。只改变选择规则，不训练新表示。',
'LightGaussian / Light4GS / Mini-Splatting / RadSplat',
'Papers/Light4GS Lightweight Compact 4D Gaussian.pdf，p5 §IV-B Eq9；Papers/Mini-Splatting Representing Scenes with a.pdf，p9 §4.2；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，pp3–4 §3.1。',
r'''变量采用S3。每种规则分别在static/dynamic组内选最高分Ks/Kd，不跨组重分配。

六个预注册行：R0=seed0均匀无放回；R1=mean_t base opacity at t（包含动态envelope，不用logit）；R2=mean_t s_it；R3=m_i=max_(v,t,p) w_ir；R4=STP迁移：mean_r 1(hit_ir)*T_ir*o_i(t)*gamma_i(t)，o是投影前激活opacity，gamma=min(volume_i/q90(volume_all),1)^0.1，volume=prod(scale)，q90=0时gamma=0；R5=用R2作无放回权重采样。

R5用exponential race：key_i=-log(u_i)/max(score_i,1e-30)，u∈(0,1)，从小选；先正分，正分不足则零分stable ID补齐；全零退化R0并标明。R4命中由真实alpha阈值后ray参与记录，不用radii。R4是明确定义的Ex4DGS迁移，gamma的q90与指数是本项目固定选择，不宣称逐行复现所有版本Light4GS。

伪代码：load M0,K1 → scores per row → per-kind select → gather all fields → serialize → S5评测。R2/R3不因多算指标增加采样。''',
'R0每组预算和无重复；R5小型3点Monte Carlo验证抽样频率随权重增加且无替换；opacity用sigmoid后值；R2以CPU逐ray累积对照；R3不是mean/max_time的误写；R4 q90仅模型与训练统计，不看V。',
'S0/M0，b=.5，seed0，六行=6run，均0FT；依次R0→R1→R2→R3→R4→R5。以后表示赛道需要的4个字节匹配对照由P99调度附属P01/P02，不额外搜索。',
'这是必须保留的参照，不按收益淘汰。R2必须完成才能运行大部分复杂方法；如R4统计未实现，其余继续且R4标BLOCKED_CACHE。选择V最好的简单行用于阶段2，但必须保留全部六行失败/结果。',
'只测到贡献不代表真实删除误差；R4 volume权重可能偏大点。数值原理是排序/加权采样，渲染有效性是经验假设。', 'K0,K1','低','1',6)

add(2,'counterfactual_error','单删除反事实误差迁移','A',
'强贡献高斯若颜色与后景相似，其真实删点误差仍可很小。观察w排名和删除失真排名错位，检验颜色/遮挡信息是否优于P01。',
'GaussianPOP；Speedy-Splat sensitivity',
'Papers/GaussianPOP Principled Simplification Framework for.pdf，pp3–4 §3.2/3.3 Eq4、Alg1；research/qa/pop-03.png可核对平方范数。',
r'''对固定原模每ray从后向前重建b_after（包括背景），d_ir=T_ir*a_ir*(c_ir-b_after_ir)。E_i=mean_(v,t,p,RGB)(d_ir²)，按E降序保留。用后缀复合而非(C-P)/(T+eps)，避免T极小时除法放大；原alpha cap和clip按S3。所有d为有符号RGB向量，先平方再累积，不把各ray向量先相加。

理论：固定深度/属性、不改变剪裁规则的单删颜色差恒等式；实际CUDA早停/边界分支导致近似，必须S3验证。多删后的E并不加性，P07/P08另诊断。本任务无重估、无FT、无GT加权。

伪代码：K2→E=sum(e_it)/24→stable topK each kind→save/eval。K2失败时不得用sum(w²)顶替并仍叫P02。''',
'在2–5点CPU ray含同色、完全透明、前后互换、黑背景/非黑背景测试中，解析d与逐个删除重合成一致；32个真实候选ID真实重渲染验证；同色前后景d应为零，即使w很大；不要取color norm代替差色。',
'S0/M0，b=.5，seed0，1行1run，0FT；先32ID×2视时单删probe，上限10min计该run统计预算，再完整选择。',
'比较P01六行；若真实单删匹配但联合质量不佳，归因于joint interaction待P07/P08，不认定公式错。作为強基线保留到后续；研究增量不能只比较opacity。',
'全ray列表显存/CPU内存高；稀疏采样不覆盖短事件。K2近似失效属于代理/实现问题，非方法负结果。', 'K2','中','2',1)

add(3,'temporal_aggregation','时间聚合与短暂事件诊断','B',
'时间平均会稀释短暂但不可替代的作用；比较mean、max、上尾均值，观察短可见点是否被救回。三个聚合属于一个诊断，不是三个创新。',
'TC3DGS平均mask；RadSplat最大贡献；Light4GS时间累积',
'Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7 Eq5/6；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4 score表。',
r'''输入固定K2的e_it≥0，先每t平均视角/像素，再在24个t聚合：R0=mean_t e_it；R1=max_t e_it；R2=mean最高ceil(.1*24)=3个e_it。三者保留最高分，时间未见必须0而非缺失跳过；最大值只有一个time也允许。不得加入mean与max混合权重。

机制标签：support_i=#{t:e_it>0.05*max_t e_it}，max=0则support=0；短暂=1..3，持续≥12。报告按标签组的保留概率、被删点E分布及其投影区域新增误差。不是语义标签，不把检测到的motion ROI当人工真值。

伪代码：e[N,24]→三个row score→同Ks/Kd→比较；额外以T24奇偶各12个时间计算排名稳定性（只诊断，不产生第四选择模型）。max/CVaR是风险偏好，不能套单删平均loss最优性证明。''',
'构造单尖峰与恒定序列使三规则产生预期不同排序；全0不NaN；量纲相同；每视角缺帧触发manifest错误；检查采样遗漏尖峰时任何聚合都无能为力。',
'S0/M0，b=.5，seed0，3行≤3run（R0=P02若hash相同直接复用，仍占上限槽），0FT，顺序mean→max→tail；追加固定窗口W时间评测。',
'若P04/P10不胜这里的简单聚合，复杂机制收益可被简单解释。短可见标签无尾部质量关联时不继续构造复杂“重现保护”；先记采样aliasing或数据事件不足。',
'极端max对噪声/单视角异常敏感；24点没有短事件保证。不能宣称时域稳定性仅靠低最差帧MSE。', 'K2','低','3',3)

add(4,'saturating_coverage','饱和时空覆盖集合选择','C',
'独立排序会把预算耗在同一时空patch，留下局部空洞；集合边际覆盖可以在相同K下补足。观察覆盖缺口是否预示真实新增误差。',
'SafeguardGS / Mini-Splatting；Light4GS；经典最大覆盖',
'Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4 §3.2；research/sources/Lazier_Than_Lazy_Greedy.pdf，p3 Alg1/Theorem1；research/literature_matrix.md关键推导1。',
r'''A_ui、patch权重w_u按S3。tau_u=0.5*sum_i A_ui；sum=0的行删除。F(S)=sum_u w_u min(1,sum_(i∈S) A_ui/tau_u)。理论条件A≥0、tau固定、w≥0，F为单调次模。真实渲染不是F。

两个行：R0=饱和F；R1=相同A/tau但不饱和的modular score sum_u w_u*A_ui/tau_u。R1用来检验收益是否只是patch归一化。tau=.5是固定经验设计，不进行阈值网格。

组配额：先在static内用Kd=0目标选Ks，再在dynamic内以已选static为底集选Kd；另用微型oracle评估次序影响。每组采用stochastic greedy epsilon=.05，每一步从剩余候选均匀无放回采样ceil((N_group/K_group)*log(20))个，按真实F边际选一个，ties按ID，selected永不重复。K=0跳组。报告实际F/evals；剩余边际全0时按P01.R2补足该组预算。此组顺序约束下不声称整体1-1/e保证；该界只适用于每个固定底集的单组基数子问题。缓存CSR，不枚举N×U。

伪代码：coverage=0→for group→for k→sample IDs→delta from their sparse columns→argmax→update coverage。不得用一次性singleton排序冒充greedy。''',
'N≤12穷举最优集合、验证边际递减及lazy/stochastic目标值；两个同patch高分点与独占patch点例子必须可改变决策；tau为0处理；组合保留全点时一致；比较16/64ray的目标值与真实render偏差。',
'S0/M0，b=.5，seed0，2行2run，0FT；先1%点最多1000次增量smoke测CPU代价，再全规模。超过45min selection保存部分进度并FAILED_COST，不换成topK蒙混。',
'需胜P01最佳/P02/P03及R1，且覆盖缺口与真实ΔMSE有正相关；如仅F提升而image/tail不变，标PROXY_INVALID或NEGATIVE。简单归一化解释全部收益则降格工程基线。',
'稀疏采样miss、冻结T、饱和阈值、组顺序；理论近似界与PSNR无关系。全量greedy的墙钟可能高于渲染节省。', 'K3','中，CSR上限8GB','4',2)

add(5,'trajectory_prototypes','轨迹原型代表性选择','B',
'具有近似运动、形状与颜色的完整轨迹形成重复簇，选择代表点可能优于按patch质量分配预算；检验几何/运动相似是否真对应可替代性。',
'OMG4聚类；GHAP；设施选址',
'Papers/OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING.pdf，pp6–7 §4.3；Papers/Gaussian Herding across Pens An Optimal Transport.pdf，摘要仅用于定位；research/sources/Lazier_Than_Lazy_Greedy.pdf，pp2–3。',
r'''从K0在t={0,74,149,224,299}取位置，除以原模型mean位置bbox对角L（L=0设1）；附加RGB DC（SH2RGB），logscale三维。z_i∈R^21，各维用所有原点median和IQR标准化，IQR<1e-6设1；clip[-10,10]。按kind分开，为每点建立t=149位置的16个最近邻（CPU分块空间索引），并含self。h_ij=exp(-||z_i-z_j||²/21)，不在邻接则0，h_ii=1。

需求权重a_i=P01.R2_i/sum_group score，sum=0用uniform。目标F(S)=sum_i a_i max_(j∈S) h_ij，empty max=0。每组stochastic greedy epsilon=.05选择Ks/Kd。R0=该facility location；R1=相同a与邻域的singleton score sum_i a_i*h_ij排序，控制“仅邻域平滑”。

它优化原轨迹代表性而非P04的image-space mass coverage；不新增/合并Gaussian属性。设施选址是已有数学结构，定位为B诊断，不能把换相似度说新算法。截断kNN使真实相似丢失但仍保非负submodular目标；不保证alpha等效。''',
'相同轨迹但颜色不同的toy分开；不同轨迹交叉中点不得被仅t149邻域误认完全相同；self边确保全保留F=sum a；标准化无NaN；kd-tree距离候选对小集暴力核对。',
'S0/M0，b=.5，seed0，2行2run，0FT。缓存21维float32约21MB，不建立N²距离；CPU select≤45min。',
'与P01/P02/P04比较；若几何代表性提升但图像变差，证明近邻相似不充分，可为P13合并提供负证据；若R1解释全部收益，停止facility优化扩展。',
'相似度经验、固定5时刻aliasing、邻域可漏快运动；不是新的OT或覆盖理论。', 'K0,K1,K4','低','5',2)

add(6,'worst_time_allocation','最差时间段约束的预算分配','C',
'平均最优的集合可能系统牺牲某个时间段；显式提高最差段保留效用能够减少尾部质量崩溃，而不是只保单点峰值。',
'Light4GS时间汇总；TC3DGS时间一致性；robust allocation',
'Papers/Light4GS Lightweight Compact 4D Gaussian.pdf，p5 Eq9；Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7；research/literature_matrix.md关键推导1。',
r'''将24个时间按索引连续分为6段，每段4时刻。E_ib=sum_(t∈b) e_it；z_b(S)=sum_(i∈S)E_ib/(sum_i E_ib+1e-12)，分母为0的段从目标去掉。目标max_S min_b z_b(S)，同组配额。此min一般非次模，不能套P04保证。

R0：multiplicative weights固定5轮近似。lambda_b=1/B；每轮score_i=sum_b lambda_b E_ib/(total_b+eps)，按组topK；记录S_j、z；lambda_b←lambda_b*exp(2*(1-z_b(S_j)))再sum归一。最终从5个候选中选min_b z最大者，tie选mean z再ID。这是代理内选择，不看V；5轮只最终一个正式run。R1：lambda均匀的一轮topK，同样归一化，对照“仅分段归一化”。

观察每段贡献保留率与V上分段ΔMSE是否匹配；另外比较P03峰值法。全零时返回P01.R2。''',
'六段极不平衡toy应重新分配；指数稳定（先减max loglambda）；统计总和与全点z≈1；保存五轮候选及objective，最终选择不读V；检查逐组配额。',
'S0/M0，b=.5，seed0，R0/R1两行2run，0FT；5轮不渲染，禁止每轮V挑选；额外窗口W。',
'需提升真实最差段/最坏10%且不过度伤mean（S7）；R1/P03同样好则robust约束没有独立贡献。代理z与真实段误差无关先判PROXY_INVALID。',
'5轮无全局解保证；以全模E固定处理忽略交互；min目标可过拟合最难段。', 'K2','低','6',2)

add(7,'iterative_recalibration','删点后重估的因果对照','A',
'删去遮挡点后剩余点的重要性变化，一次性排名过时；逐轮重估是否解释复杂集合方法的收益。',
'GaussianPOP iterative re-quantification；PUP多轮',
'Papers/GaussianPOP Principled Simplification Framework for.pdf，§3.4（执行只需重新阅读该节）；Papers/PUP 3D-GS Principled Uncertainty Pruning.pdf，p4 §4.4。',
r'''分4轮，从原组数n_g到k_g，轮j目标n_gj=n_g-floor(j*(n_g-k_g)/4)，j=1..4。R0每轮在当前模型重新计算P02.E，再删最低到目标；R1用原始E进行相同分批删除，且每轮也重算E用于诊断但不用于决策。这样两行统计成本相同。无中途FT，无随机重采样，C4/T24/尺寸完全相同。

每次缓存以parent model hash隔离，参考M0图像持续作为teacher质量对照，但单删E的teacher是当前轮模型。保存每轮E在共同ID上的Spearman、排名变化、实际新暴露ray数。

伪代码：M=M0; for j: collect K2(M); select using fresh/stale; apply; save intermediate; 最后eval。不把四轮当四个研究方案。''',
'R1最终IDs必须与P02单轮完全相同；不同轮hash不同；新暴露toy中第二点贡献应提升；没有隐性微调、opacity reset或补点；每轮原始ID保持。',
'S0/M0，b=.5，seed0，2行2run，0FT；每行4次统计计入45min run上限。先2帧4轮smoke，如统计估计超预算只标成本阻塞，不改两轮后称同法。',
'胜P02且新的可见性变化预测改善可支持重估；若P04不胜重估，应将其定位为省统计成本近似而非更强机制。首次崩溃先核查早停replay和cache。',
'统计成本可能主导；多轮独立误差仍非联合最优，不能引用原文含FT收益到本无FT实验。', 'K2，每轮失效重建','中','7',2)

add(8,'joint_deletion','联合删除的颜色与遮挡交互','C',
'两个各自安全的删除组合可能不安全，或者其颜色误差互相抵消；明确联合干预应改变边界点选择。',
'GaussianPOP单删；PUP忽略跨点block；MaskGaussian自适应',
'Papers/GaussianPOP Principled Simplification Framework for.pdf，p3 Eq4；Papers/PUP 3D-GS Principled Uncertainty Pruning.pdf，p4 Eq8–10；research/literature_matrix.md推导2。',
r'''以P02组内topK为初集S。每组取保留边界最低32点与删除边界最高32点（不足取全部），共≤64候选/组，其余点固定。用K3采样ray保存原完整alpha有序列表，以移除候选及固定已删点后的真实重放计算D(S)=mean||I_S-I_M0||²，包含背景、重新算T，不能冻结原T。

R0=交互交换：最多16次交换/组。每步枚举candidate内所有(保留a,删除b)交换，分块评估同ray replay D(S-a+b)，只重放交换两点涉及的ray，其余ray的D为固定常数；选最小且严格改善>1e-10的一对；更新S并重放。每步每组最多32²交换，GPU逐batch16对，选定后验证真实2帧render（每4步一次，最多4次/组）。若预计成本超45min按原候选大小停止，不临时缩候选取得成功。R1=相同候选/迭代上限，但用sum_(deleted) E_i加性目标交换；理论上P02已最优，通常零交换；同样执行诊断replay计公平冷启动成本。

辅助报告32固定邻近候选对的I_ij=D(delete i,j)-D(delete i)-D(delete j)与signed-d内积近似；这是诊断不是新的调参。这里非submodular，无全局保证。''',
'2前景同色点前后遮挡toy：独删低误差但双删高误差；正负颜色误差抵消toy；对≤8点穷举检查exchange目标；swap保持kind数量；replay对真实render误差按S3门槛。',
'S0/M0，b=.5，seed0，2行2run，0FT；64候选/组上限、16 swaps预注册；先8点/2ray CPU和2帧GPU smoke。',
'需真实joint interaction幅度显著于数值噪声且R0真实质量优于P02/R1；如果仅采样D改善或交换0，报代理不足/局部无机会，不宣称机制有效。未来扩展只由S7决定。',
'组合渲染代价高，局部搜索非全局；忽略候选外可换点，结果阴性仅限边界交换模型。', 'K2,K3,K5','高，严格chunk','8',2)

add(9,'neighbor_recoverability','邻居补偿可恢复性选择','C',
'当前删除误差高不代表微调后不可恢复；若误差落在存活邻居颜色可表达空间，可能安全释放容量。',
'MaskGaussian补偿动机；PUP Fisher；AAAI Plug-and-Play detail compensation',
'Papers/MaskGaussian Adaptive 3D Gaussian Representation from Probabilistic Masks.pdf，p3动机/p4 §3.2；research/literature_matrix.md推导4、补充检索（AAAI仅摘要，不假装复现）。',
r'''以P02 S0为固定基线，每kind取边界两侧各256点构成Q（≤512），固定保留S_anchor=S0\Q。对i∈Q，找S_anchor内t149位置最近8点J_i；只允许保留锚点作补偿，避免循环“互相可替代”。以16-ray patch样本的实际d_i RGB向量为target，构造颜色DC Jacobian列：D_j,r=w_jr*SH_DC_const（按仓库SH2RGB导数），RGB三个通道独立，r覆盖i或邻居支持，训练ray按hash奇偶分fit/check。

每channel解ridge x=(DᵀD+lambda I)^-1 Dᵀd_i，lambda=1e-3*trace(DᵀD)/8+1e-10，把补偿引起的RGB DC变化clip[-.05,.05]后重新计算留出ray残差。R_i=mean_check||d_i-D x||²；无邻居则R_i=E_i。R0用R排序Q选足该组原Q保留数，anchors保持；R1在同Q用E排序。选择结果阶段1不实际施加x（避免偷加优化）；另做16个按原E分位等距取的候选删点+邻居DC解补偿probe，与相同20步邻居DC Adam真修复对照，仅记录恢复相关性，不用于选择。

P09预测未来有限FT可恢复性，阶段1零FT可能差；若预测经probe验证，允许以机制证据晋级，不以零FT更差直接否定。进入阶段3必须统一总1000步，不能额外给P09局部训练。''',
'线性可表达与正交target toy；fit/check严格独立；DC Jacobian有限差分；对补偿point最终确实保留作断言；ridge病态用float64 Cholesky，失败加10倍lambda仅一次记日志，否则该i回E；不能用平方score拟合有符号误差。',
'S0/M0，b=.5，seed0，R0/R1两行2run，0FT；16候选×最多20步probe合计≤320步、10 GPU分钟，计P09总预算；不能全场逐点重训。',
'主证据为check residual对真实20步恢复后的MSE排序Spearman≥.5且好于原E至少.1（决策规则）；阶段3再看最终质量。若仅近邻距离/原E能解释，降格。优化失败与线性代理不成立分开。',
'只估颜色恢复，不代表几何/轨迹恢复；clamp/alpha nonlinearity、ray subsampling与遮挡变化可破坏线性近似。', 'K2,K3,K4,K5','中','9',2)

add(10,'reappearance_guard','消失后重现的事件保护诊断','B',
'只看全时均值或峰值可能漏掉同一轨迹两段可见事件之间的保护需求；硬事件约束是否改善重现后的尾部误差。',
'USPLAT4D遮挡；TC3DGS时序mask；SafeguardGS pixel保护',
'Papers/UNCERTAINTY MATTERS IN DYNAMIC GAUSSIAN.pdf，p4 §4.1；Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7；Papers/SafeguardGS 3D Gaussian Primitive Pruning While Avoiding Catastrophic.pdf，p4。',
r'''用K1每点s_it判active= s_it≥0.05*max_t s_it且s_it>0，按24索引连续段，gap至少2个采样时刻分开，至少两段的ID为reappearing代理。它不是对象身份真值。对每个此ID的每段生成需求e=(ID,episode)。可满足该需求的候选为该ID及其K4同kind16近邻：必须在episode的每个样本t都满足s_jt>0且投影中心到i在C4中至少一相机距离≤同图8pixel，并且DC RGB距离≤.1；否则不许宣称替代。

R0从P02集合起，按episode的sum e_it降序取前min(256,floor(.1*K))需求做保护repair。若某需求无当前存活候选，插入其候选中E最高者，同时移除同kind保留集中E最低且移除不使此前已满足需求失效者；没有可移点则记infeasible，不超K。不允许后来的repair破坏先前保护。R1完全相同需求清单/成本但不repair，返回P02。

记录被保护需求数量、infeasible、回归ID重现后首个采样时间的真实V误差；固定窗口W若没覆盖重现，另输出最多5个最早重现时刻±2帧诊断（最多50图，计成本，不能替换标准指标）。''',
'两段可见且中间遮挡toy；单持续段不计重现；零分不生成需求；ID变动不让轨迹身份漂移；repair逐步验证此前constraints和K；无候选不回退随便保点。',
'S0/M0，b=.5，seed0，2行2run，0FT；若没有≥10个可行重现需求，只报事件不足，跳正式P10并继续其他任务，不人为挑test事件。',
'与P02/P03峰值/P04比较；只有被保护事件区域重现误差下降且全局门槛过才支持。若max已足够，机制复杂性没有价值。此为B诊断，不把新约束名称包装原创。',
'训练相机遮挡不能推断所有视角；24采样漏短事件、轨迹对应物体不保证，可能只是漂移/透明度变化。', 'K1,K2,K4','低','10',2)
