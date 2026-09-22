"""Document authoring only; no experiments."""
from build_prompts import ROOT, items, add
import json

add(11,'learned_budget_gate','可恢复的定额掩码优化迁移','A',
'在删除前优化集合参与度可让相关点竞争与恢复，比固定评分更接近重建目标；检验收益是否仅来自额外训练。',
'MaskGaussian / GaussianSpa / CDGS / RD4DGS',
'Papers/MaskGaussian Adaptive 3D Gaussian Representation from Probabilistic Masks.pdf，p4 Eq3/4；Papers/GaussianSpa An “Optimizing-Sparsifying” Simplification Framework for.pdf，p5 Eq6–16；Papers/Temporal Smoothness-Aware Rate-Distortion.pdf，p4 Eq4。',
r'''固定原模型属性，新增每点logit l_i，初始化logit(0.9)，加1e-3*标准化P02.E打破tie，E标准化用mean/std+1e-8。组内K固定。

R0=hard topK STE：p=sigmoid(l)，h=topK(p)的0/1，gate=stopgrad(h-p)+p；在raster内部把alpha改为gate*a且保留原a参与列表，T递推使用(1-gate*a)。alpha阈值判断用未masked a，不让h=0点丢失mask梯度。全部非gate属性冻结，Adam(l,lr=.01) 200步，batch1 C随机视时、0.8L1+.2DSSIM；每步精确K但全候选仍占显存/统计时间，最终实体删除topK。

R1=soft gate budget：相同l/200步，gate=p，loss增加10*((sum_g p-K_g)/N_g)^2对两组求和；最后hard topK，记录soft→hard质量落差。R2=P01.R2直接剪到K后对全部保留属性做200步S6普通FT（相同batch/数据抽样），作为额外训练解释对照。R0/R1只是在P11中的优化器诊断，不独立计创新。

理论：STE有偏梯度；soft数量不是精确非零点数；非凸解无全局保证。不是原版MaskGaussian的Gumbel或GaussianSpa完整ADMM复现。实现mask梯度后再开运行；禁用densify等S6操作。''',
'mask=all1与原render等价；mask0仍有有限差分方向一致的mask梯度；2点可替代toy验证一方masked仍可恢复；hard前向点数精确；验证opacity筛选未截断被mask点；梯度不写回冻结属性。',
'S0/M0，b=.5，seed0，3行3run；每行200步选择/训练计入预算，不标0FT方法；正式前10步smoke。阶段3 R0/R1再800步普通FT；R2从原checkpoint重做总1000步，不把已200步结果再加1000。',
'必须同时看P02零FT和R2等步FT，只有胜等成本控制才支持学习选择价值。soft模型好但hard坍塌属OPTIMIZATION_FAILED，不追加训练到成功。作为强迁移基线保留。',
'掩码CUDA反传改动、筛选/早停梯度、Windows扩展编译风险最高；不能用普通opacity乘法伪装完整恢复mask。', 'K2,K6，训练后全失效','高','11',3)

add(12,'static_dynamic_allocation','静动态容量分配诊断','B',
'原始静动态数量比例可能不是压缩后的最佳比例；改变容量分配而不改变组内评分可解释部分动态保护收益。',
'CDGS adaptive allocation；Ex4DGS / Swift4D',
'Papers/Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf，pp4–6；https://arxiv.org/html/2602.03538v1 §III-C（在线不可达用literature_matrix概述，不猜原实现）。',
r'''按P02.E组内排序，仅变更Kd。三行：R0=原比例floor(K*Nd/N)；R1=global topK E自然确定Kd（取消kind配额）；R2=风险均衡分配。R2候选Kd=round(q*K)，q∈{0,.1,...,1}再clip到[max(0,K-Ns),min(K,Nd)]并去重，Ks=K-Kd。每候选计算保留static E比例z_s及dynamic E比例z_d，取min(z_s,z_d)最大，tie取保留总E最大再较小Kd；不渲染中间候选、不看V。总E=0组从min目标删除，双0回R0。

R2是离散代理容量分配，不是CDGS训练中的learnable controller。三行严格K，bytes不同，报告实际bytes并按bytes画Pareto。不能把较高dynamic成本换来的质量提升称同资源胜利。''',
'group容量/clip；global topK与两组排序合并一致；零E无除零；11候选是代理比较，不产生11个V调参run；总K恒定。',
'S0/M0，K=floor(.5N)，seed0，3行3run，0FT；先报K匹配，再用P19审查byte匹配。',
'若复杂方法收益伴随动态点增多，必须补本控制；只有count下优而byte下无优势，结论限于预算单位选择。B无原创主张。',
'跨组分数可比依赖相同ray平均；某组遮挡会使E不准；较贵动态容量不等部署预算。', 'K2','低','12',3)

add(13,'trajectory_merge','轨迹一致的矩匹配合并迁移','A',
'直接删除丢掉局部质量，两个近似轨迹合为一个能保留位置/形状信息；检测何时合并比保一删一更好。',
'OMG4 Gaussian Merging；GHAP mixture reduction',
'Papers/OPTIMIZED MINIMAL 4D GAUSSIAN SPLATTING.pdf，pp6–7 Eq3–5；literature_matrix推导5。Eq3原印正色差项与文字相似性不符，本任务不复现该式。',
r'''原M0每kind找t149位置16近邻，候选无重叠pair。距离d_ij=max_(t∈T24)||x_i(t)-x_j(t)||/L + ||RGB_DC_i-RGB_DC_j||；L=原场景bbox diagonal，零则1。按d升序/ID tie贪婪配对。只允许颜色差≤.1、max位置距离≤2*max_i,j(mean scale)，且动态opacity envelope在全部300整数时间的max abs差≤.1；不跨kind。

权重h_i=E_i/(E_i+E_j)，双0用.5。每时间mu=h_i*x_i+h_j*x_j；Sigma(t)=sum_l h_l[Σ_l(t)+(x_l-mu)(x_l-mu)^T]。兼容动态scale恒定：取T24平均Sigma做eigen分解，eigenvalue下限1e-8*L²；其scale/rotation在所有keyframe固定（会损失原旋转，须报告）。静态mean/disp做相同加权；动态position所有原keyframe组合。SH逐系数加权；base opacity=1-(1-o_i)*(1-o_j)，clip[1e-6,1-1e-6]转logit。动态envelope取E较高父点，tie ID。不能擅自发明时变scale。

R0从最相似pair依次merge到组K或无可行pair；若未达K，用P02删除最低E单点补预算并记录merge贡献数量。R1相同pair顺序，每pair只留E高父点，其余补预算相同。合并E以父E和仅作fallback排序，不称新真实贡献。无FT。''',
'同位置/协方差时moment一致；Sigma PSD、旋转正交/quat规范；opacity有限；父ID不重复合并；全保留identity；ray颜色不守恒，toy展示误差。',
'S0/M0，b=.5，seed0，2行2run，0FT；32pair×2帧probe再全模型；可行pair<1%N报机会不足，不放宽阈值。',
'R0胜R1/P02且实际合并比例充分才支持；fallback剪枝收益不算合并有效。明确是迁移控制，不是完整OMG4复现。',
'透明度非线性、轨迹交叉、时变cov受限；父ID管理。引入时变scale属于新研究决策，不临场扩展。', 'K0,K2,K4','中','13',2)

add(14,'dynamic_to_linear','动态轨迹向线性静态表示回收','C',
'训练时归为dynamic的点在最终模型中可能仅需线性位移与固定旋转；以误差证据回收轨迹存储可能比删整点安全。',
'Ex4DGS分解；Swift4D / CDGS静动态分配',
'Papers/Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf，p4 Eq5、pp5–6 opacity；scene/c_gaussian_model.py:get_static_xyz_at_t。',
r'''对每dynamic在所有原keyframe真实时间（含padding）LS拟合x(t)=a+(t/300)b。rotation取t149 normalized quaternion，scale/SH保持。转换候选要求全部300整数时间opacity envelope range≤.01且min≥.99，rotation角距离max 2acos(abs(dot(q(t),q149)))≤2度。位置误差R_i=mean_(T24) s_it*||x_i(t)-xlinear(t)||²/(mean_t s_it+1e-12)/L²。

R0按R_i/bytes_saved升序转换；R1相同候选按max_t||x_i(t)-mean_t x_i(t)||/L（运动幅度）升序。停止于表示赛道目标或候选耗尽，实体N不变。转换_xyz=a,_xyz_disp=b,rotation=q149,opacity=base；不把b设0。实际saved含kind属性和ID元数据，不假设固定1KB。

不为达预算越过opacity/rotation gate；未达输出可达RD点及BLOCKED_TARGET。LS几何精度不保证image精度。新颖潜力仅在有证据的posthoc转换决策，静动态分解本身已知。''',
'原static轨迹采样应LS精确恢复；非线性往返净位移0不能当静止；短opacity拒绝；padding真实时间；转换后kind变而parent ID保留。',
'S0/M0，seed0，2行2run，N不变，目标B_fixed+.5B_variable（S4），0FT；B_variable=全动态比转换为static多出的payload。',
'需胜同bytes删点且好于简单幅度排序；极少可转换则记录已充分分离，不降低保护门槛。',
'opacity/rotation短暂变化、position投影敏感；微小几何误差不等小图像误差。', 'K0,K1,K4','低','14',2)

add(15,'adaptive_keyframes','自适应轨迹关键帧迁移','A',
'统一关键帧密度在平缓轨迹浪费字节；误差驱动分配可能保急转弯而减少静缓区存储。',
'TC3DGS keypoint interpolation；Ex4DGS keyframes',
'Papers/Temporally Compressed 3D Gaussian.pdf，p8 Alg1；Papers/Fully Explicit Dynamic Gaussian Splatting — Ex4DGS.pdf，p5 Eq6–9。',
r'''原P_i∈R^(J×3),Q_i∈R^(J×4)，J由PLY读取含padding。新文件保存per-ID joint knot indices、P/Q，端点强制；仅压动态position/rotation。解码缺失P线性插值、Q shortest-arc slerp，恢复原J dense keyframes，再调用原cube/slerp renderer。Q先normalize，相邻dot<0取反；全保留模式不改原tensor。

R0每点首末knots起步，缺失k的epsilon_ik=||P_ik-Pinterp_ik||²/L²+(angle(Q_ik,Qinterp_ik)/pi)²。每次将全场最大epsilon的k加入对应点，更新该点误差；ties ID,k。仅在actual bytes能容纳新k时加入，达到目标或全误差0终止。R1 uniform：全场轮转按ID，逐点最长时间区间二分添中点（并列较早区间），同byte目标。索引uint16（J>65535则uint32）及每点offset计入bytes。

这是TC3DGS思想迁移，两误差项等权为固定经验选择。decode dense只省磁盘，不预称GPU/速度收益。原cube可能放大knots误差，需测过渡帧。''',
'线性/常旋转只需端点；q和-q等价；180度slerp核验；padding knots范围；allknots decode一致；t=.5真实render。',
'S0/M0，seed0，2行2run，N不变，目标B_fixed+.5B_variable，B_variable=动态P/Q payload；0FT；128轨迹CPU+2帧GPU smoke。',
'same-byte胜uniform和P01/P02；仅knots MSE改善而image恶化则代理不充分。A迁移，不声称新RDP。',
'不规则索引/外推padding/quat约定；decode dense不省GPU。', 'K0,K4','低','15',2)

add(16,'haar_motion','Haar轨迹压缩迁移','A',
'相邻关键帧有低频冗余；压缩位置细节是否比删点更划算。',
'RD4DGS §3.3',
'Papers/Temporal Smoothness-Aware Rate-Distortion.pdf，p5 Eq6/7。',
r'''仅压动态P[N_d,J,3]，其余不变。J奇数时repeat末keyframe，记原J。一级Haar a_k=(P_2k+P_2k+1)/sqrt2，d_k=(P_2k-P_2k+1)/sqrt2。R0丢全部d，只存a float32，decode每对P=a/sqrt2，截回J再原renderer。R1直接P存FP16、decode FP32（相近position bytes，无低通）。不加quant/FT帮助R0。

两行bytes由编码固定，padding略有差异；按actual bytes报告，差>1%不称等字节。P01/P02按更大actual byte cap匹配（S4共享4个资源对照槽）。全保留模式同时保a,d并逆变换验证roundoff，不能将真正压缩误差归为数值噪声。

丢高频可能带来运动相位偏移。完整RD4DGS还含联合优化，本模块负结果不否定原论文。''',
'正逆Haar、常值d=0、线性轨迹pair averaging非exact、奇数长度；FP16溢出报失败不clip；位置与位移不得混用。',
'S0/M0，seed0，2行2run，N不变，固定一级Haar/FP16，0FT；B_variable=动态P payload，约减半。',
'比较R1、P15、same-byte剪枝；FP16解释收益则Haar无必要；仅训练能恢复留阶段3统一预算验证。',
'高频运动/position精度/边界；仅磁盘收益，不能引用完整RD4DGS倍数。', 'K0','低','16',2)

add(17,'shared_motion_basis','跨轨迹共享低秩运动基','A',
'同物体轨迹可能共用运动基，跨点相关压缩能否胜逐轨迹时间编码。',
'4DGC共享motion grid；TC3DGS / RD4DGS',
'research/sources/4DGC.pdf，p4 §3.1/3.2；Papers/Temporally Compressed 3D Gaussian.pdf，p8。',
r'''只压动态P[N_d,J,3]。每点首关键帧p_i0单存，X_i=vec(P_i-p_i0)∈R^(3J)。R0 CPU分块XᵀX（3J维，禁止N²），eigh取前r个eigenvectors V，A=XV，Xhat=AVᵀ。r为使4*(3Nd+Nd*r+3J*r)+header≤target_variable_bytes的最大整数，r≤min(Nd,3J)，r<1报目标不可行。eigenvector符号按首非零分量正规范。

R1每条坐标序列正交DCT-II时间basis，保前r_t系数，r_t取满足相同actual byte cap的最大整数，三轴等长；DCT基固定解析不用存，CPU显式J×J矩阵可实现，无需安装库。rotation等不变。解码denseFP32再原renderer。

低秩理论仅保证Frobenius最佳rank-r误差，不保证图像/可见性最优；不增加exceptions残差文件突破预算。''',
'共享平移rank≤3可恢复；随机独立轨迹不应虚报低rank；full rank相对误差≤1e-5；分块Gram与小集SVD核验；大负eigenvalue报错；basis/offset成本计入。',
'S0/M0，seed0，2行2run，N不变，B_variable=动态P，目标B_fixed+.5B_variable，0FT；1024轨迹smoke。CPU≤45min，显存估计低但未实测。',
'同bytes比较R1/P15/P16；DCT解释收益则跨轨迹共享无价值。A编码迁移，不宣称新数学工具。',
'多物体全场低秩混合/绝对坐标/decode dense；速度可能更慢。', 'K0,K4','低','17',2)

add(18,'temporal_active_set','受切换约束的时间活动集合','C',
'整轨迹删除太粗，每帧独立停用可能闪烁；区间选择是否能在等活动成本下兼顾短事件和稳定性。',
'TC3DGS temporal masks；PD-4DGS；4DGC显露补偿',
'Papers/Temporally Compressed 3D Gaussian.pdf，pp6–7 Eq5/6；Papers/PD-4DGS Progressive Decomposition of 4D Gaussian.pdf，摘要；shared_protocol S5。',
r'''将0..299按最近T24时间分24个整数Voronoi bins，tie归较早bin，长度n_b。z_ib∈{0,1}，目标sum_i,b n_b*e_ib*(1-z_ib)+lambda*sum_i,b n_b*z_ib+gamma*sum_i,b>0|z_ib-z_i,b-1|。活动Gaussian-frame cap=.5*N*300。R0 gamma=.1*median_{e_ib>0}(n_b*e_ib)，全0取0；R1 gamma=0，其余一致。

每点用两状态DP精确解给定lambda的链；lambda在[0,max e+2gamma/min n_b]二分20次，分别对static/dynamic原比例cap，保存最大活动数的feasible解，ties代理失真较小；全0inactive，DP ties优先inactive再少切换。这是Lagrange relaxation近似，若预算利用率<95%标budget_gap，不超预算追质量。

实体存全部N和RLE区间，渲染每t在raster前gather active，保持kind/ID map。峰值显存可能不减，磁盘可能增；测gather后的端到端延迟。hard切换不加额外fade；S5时间评测必须执行。''',
'N≤3,T≤5穷举DP目标；0/299覆盖；RLE一致；gamma0与独立threshold一致；allactive render一致；concat ID不乱。',
'S0/M0，seed0，2行2run，实体N不变，active-frame≤.5N*300，0FT；V×T24+W及最早5个switch±2帧诊断，不替换标准指标。',
'同活动预算R0比R1 TDE降≥10%、平均质量门槛过；实际端到端加速才有部署主张。raster快但gather后慢则部署失败。',
'粗bin proxy/hard边界/DP CPU成本；保存全部点不省显存。', 'K2,K6','中','18',2)

add(19,'resource_budget','按实际资源成本选择','C',
'动态轨迹贵、大投影点慢；显式成本选择能否在等bytes或tile预算下保更多信息。',
'LightGaussian / Speedy-Splat / RD4DGS / CDGS',
'Papers/Speedy-Splat Fast 3D Gaussian Splatting with.pdf，pp4–6 §4.1/4.2；project_facts PLY成本；shared_protocol S4/S5。',
r'''固定P02 E。R0 byte cost=单点该kind实际payload+ID bytes，header单计，cap=.5B0；R1 tile cost=mean_(C4,T24)实际CUDA tiles_touched，cap=.5sum cost；R2单位cost，cap=.5N即P02。R0/R1按E/max(cost,eps)降序扫描可装点，不设kind配额。cost0且E>0优先，0/0末尾。

还比较single best feasible E点集合与greedy集合，选sum E较高者；不声称非加性渲染近似保证。byte输出超cap时按最后低E/cost点依次删至满足，记录估算误差；tile代理不因真实延迟不好而临场重调。额外同cap P02控制按E降序装入相同cost cap（资源匹配槽，不能当同K）。

R1只优化tile代理，不等同毫秒。S5测峰值/端到端。P04 coverage/cost组合不在此任务叠加。''',
'不同cost toy决策改变；0cost不无限循环；header>cap不可行；bundle所有解码文件计入；tiles_touched对forward核对；3块GPU timing。',
'S0/M0，seed0，3行3run，0FT；三个预算单位分列。额外正式resource-match评测占全局4槽，不无限增run。',
'比较same-byte/tile P02及P12；真实资源≥10%改善且质量门槛过才支持。若只是静动态比例收益，降低贡献表述；没有公平匹配对照先不晋级。',
'allocator/深度排序/early-stop非线性；ratio greedy不能保证真实最佳。', 'K0,K1,K2','低','19',3)

add(20,'attribute_rate_allocation','按视时敏感度分配SH精度','A',
'各点高阶外观需求不同，逐点SH精度分配是否优于统一降阶。',
'RD4DGS SH mask；TC3DGS mixed precision；LightGaussian',
'Papers/Temporal Smoothness-Aware Rate-Distortion.pdf，p4 Eq5；Papers/Temporally Compressed 3D Gaussian.pdf，p7 §4.1.2。',
r'''几何/opacity/轨迹/DC不变，仅SH AC。option∈{degree0,degree1,degree2,degree3_fp16,degree3_fp32}；前3保对应degree RGB系数FP32，其余0；FP16仅AC量化。固定原模ray计算delta_c_io=evalSH(original)-evalSH(option)，D_io=mean_r||w_ir*delta_c_io||²。cost=实际option payload+1byte tag；option成本不必随degree递增，先按cost去除被同时更低cost/更小D支配的选项。

R0每点最便宜option起，priority queue按升级(D_old-D_new)/(bytes_new-bytes_old)最大依次升级，ties ID；目标AC payload .5倍，含tags/offsets，不可超支；全D相等不升级。R1统一option，选cost≤同cap且sum D最小者，不用V挑。D是单点颜色变化精确项，多点平方有交互，greedy无全局render最优保证。

序列化ragged bundle，decode补零恢复dense SH+原raster。高阶全零却仍完整写PLY不算压缩；dense decode不省GPU。只用C4方向，不读V方向拟合bits。''',
'SH eval与CUDA3方向一致；degree3_fp32 D=0；DC常色不应改变；FP16 overflow失败不clip；Pareto清理成本非单调；all-original模式identity。',
'S0/M0，seed0，2行2run，N不变，B_variable=全部SH AC payload，target=B_fixed+.5B_variable，0FT；1024点统计smoke。5options是代理选择，不产生5×budget渲染网格。',
'胜uniform和same-byte P01/P02才支持adaptive；新视角高光失真说明统计泛化局限，不用test重新分bits。A迁移，不称新率失真理论。',
'视角覆盖/颜色相消/混合precision；当前SH不随时间变化，勿新增不存在参数。', 'K0,K1,K3','中','20',2)

def generate():
    for d in items:
        pid=f'P{d["i"]:02d}'
        doc=f'''# {pid} {d['name']} — Sol 独立执行 prompt

## 1. 任务目标与范围

类型 **{d['kind']}**。{d['hypothesis']}
最近工作：{d['related']}。本任务的实质区别由§3定义；不能改成其它论文完整pipeline。C只表示待验证增量，A/B不包装创新。

## 2. 必读上下文

工作目录 `E:/4DGS`。先读 `research/shared_protocol.md` S1–S8、`research/project_facts.md` 接口/模型/数据泄漏段，以及 `research/execution_order.md` 本任务依赖。必要文献：{d['refs']}
文本镜像为 `research/paper_text/<同PDF名去.pdf>.txt`，用 `===== PDF PAGE n =====` 定位。无需重读全部文献；公式提取疑问对照原PDF，不猜符号。

## 3. 精确方法定义

{d['method']}

通用边界：N、K、组配额、采样与基础统计按S1–S4；K取floor，ties按stable ID，空组跳过，epsilon仅用于明确的零值边界。NaN直接报错。经验参数已固定，不临场搜索。理论条件/近似限制不得省略。

## 4. 实现位置与接口

真实入口：`scene/c_gaussian_model.py` 的 `get_xyz_at_t/get_opacity_at_t/get_features/get_scaling/get_rotation_at_t/save_ply/load_ply/prune_points`；`gaussian_renderer/__init__.py:render`；`scene/__init__.py:Scene`；`render.py:render_set/render_sets`。先确认P00 adapter通过；prune_points依赖optimizer且True表示删除，纯推理不能盲调。统计扩展在 `submodules/diff_gaussian_rasterization_df/cuda_rasterizer/forward.cu`；需反传时审计同目录backward和Python绑定，不杜撰未核验函数。

拟新增 `research_impl/methods/{pid.lower()}.py`、`configs/research/{pid.lower()}.json`，用P00 registry/S2接口读取不可变model/cache，返回SelectionResult或表示变换、实际预算/ID映射。这些是设计接口，当前尚未实现。依赖缓存：**{d['cache']}**。属性/公式/输入变化按S3失效规则处理。

## 5. 正确性检查

先执行S2全保留一致性、ID/索引、save-load、空组、确定性检查。特有检查：{d['checks']}
toy与独立oracle发现真实错误，不只复写公式作测试。变换roundoff单列，不能把压缩失真当底噪；真实场景smoke不可省略。

## 6. 最小运行矩阵

{d['matrix']}

阶段1配置行上限 **{d['n']}**，不再乘参数网格。正式质量图1352×1014，先2帧smoke实测显存/耗时。实现成本评估 **{d['cost']}**；风险：{d['risk']} 这些评估未实测，不承诺24GB必可运行。单run≤45 GPU分钟（含方法新增统计、probe、优化、评测），选择CPU≤45分钟；smoke≤10 GPU分钟，OOM分块两次后停止，不降评测分辨率。阶段2/3仅按S7队列名额扩展，不自行跑所有场景。

## 7. 对照、验收与否定条件

{d['evidence']}

简单替代解释必须写入报告。记录actual K/bytes/活动成本、统计采样数/优化步数/耗时，不偷偷增加资源换提升。S7门槛是项目决策规则，不是统计显著性。首次失败按实现→代理→优化→真实负结果分类，允许一次规定修复机会。数学及近似风险：{d['risk']}

## 8. 输出与报告

按S8每行独立目录：resolved_config、provenance（git+diff/model/cache/data hash）、实际commands、stdout/stderr、status、逐帧CSV、summary、resources/timing、可重载模型、ID/变换映射、固定可视化。另存本方法score/candidates/预算轨迹/正确性检查。method_report分开写观察事实、近似、简单解释、公平性、失败分类与晋级资格。NOT_RUN、FAILED/BLOCKED、COMPLETED、NEGATIVE分别标；未跑不填预期指标，不自动宣称新颖性或论文级成功。

## 9. 自主执行与边界

当用户把本文件交给Sol并授权统一执行批次，可实现、修复普通代码问题并在以上预算内串行运行。Astra本轮只写文档，未启动这些实验。遵守S7/S8总预算、一次修复、GPU锁和progress断点恢复。方法冲突、关键数据缺失或须改变假设时记录具体问题/已试动作/影响依赖，标BLOCKED并继续独立任务；不临场换定义。不改原数据、参考checkpoint或既有结果，不降低评测标准使任务通过；最终价值/新颖性判断归Astra。
'''
        (ROOT/'prompts'/f'{pid}_{d["slug"]}.md').write_text(doc,encoding='utf-8')
    lines=['# 实验索引','','20个方案；A=基线/迁移，B=诊断，C=待证实候选。P00/P99不计入。公式、文献范围、最小矩阵和证伪条件见各prompt。显存均为未实测风险。', '', '| ID | 方案名称 | 类型 | 核心假设 | 最近相关工作 | 潜在贡献 | 实现成本 | 预计显存风险 | 依赖缓存 | 验证顺序 | 停止条件 |','|---|---|---|---|---|---|---|---|---|---|---|']
    stops={1:'基线必须保留；统计错误停依赖',2:'单删验证不符停K2',3:'无短事件证据或简单聚合足够',4:'F提升不改善真实误差/超时',5:'代表性无图像收益',6:'最差段无改善/mean损伤超标',7:'重估无排名/质量收益或统计超时',8:'交互弱/交换无真实收益',9:'恢复预测不优于原E',10:'重现需求少于10/峰值已解释',11:'hard坍塌/梯度错误/不胜等步FT',12:'收益仅来自更多昂贵动态容量',13:'合并机会<1%N/不胜删父点',14:'合格转换不足/不胜小运动控制',15:'不胜uniform/解码错误',16:'不胜FP16/运动失真',17:'不胜独立DCT',18:'TDE不降/端到端不加速',19:'资源代理不对应实测/不胜同cap',20:'不胜uniform/高光泛化失败'}
    for d in items:
        pid=f'P{d["i"]:02d}'
        gain={'A':'迁移强基线','B':'隔离机制与简单解释','C':'可能的机制增量，未证明原创'}[d['kind']]
        risk='高' if d['i'] in [8,11] else '中' if d['i'] in [2,4,7,9,13,18,20] else '低'
        lines.append(f'| [{pid}](prompts/{pid}_{d["slug"]}.md) | {d["name"]} | {d["kind"]} | {d["hypothesis"].split("；")[0]} | {d["related"]} | {gain} | {d["cost"]} | {risk} | {d["cache"]} | {d["order"]} | {stops[d["i"]]} |')
    lines+=['','## 阶段1运行数','','| ID | 参数行上限 | 额外优化 |','|---|---:|---|']
    for d in items:
        extra='200步，含等步控制' if d['i']==11 else '16候选probe≤320步，仅诊断' if d['i']==9 else '0FT；内部搜索/统计计时'
        lines.append(f'| P{d["i"]:02d} | {d["n"]} | {extra} |')
    lines+=['',f'方法行合计 **{sum(d["n"] for d in items)}**。另4个共享资源匹配对照槽=**51**。按首次需要的实际byte/tile cap建立P01/P02对照，后续同key复用；超过4种新cap则BLOCKED_BUDGET，不声称公平强对照齐全，阶段2额度内再补。共同未剪基线另最多9run。阶段2≤96、阶段3≤108，正式评测run≤264；另24次失败修复槽。见shared_protocol S7。','','P01六分数、P03三聚合、P11两mask优化都是内部控制，没有独立凑方案。P04图像覆盖、P05轨迹代表性、P06最差时间段、P10硬事件保护分别有不同失败假设。P14类型转换、P15稀疏knots、P16固定小波、P17跨点共享、P20外观精度操作不同信息结构，均不重复申报新颖性。']
    (ROOT/'experiment_index.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'experiment_manifest.json').write_text(json.dumps([{'id':f'P{d["i"]:02d}','name':d['name'],'type':d['kind'],'prompt':f'prompts/P{d["i"]:02d}_{d["slug"]}.md','stage1_rows':d['n'],'cache':d['cache']} for d in items],ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote',len(items),'prompts; stage1 method rows',sum(d['n'] for d in items))

if __name__=='__main__':
    generate()
