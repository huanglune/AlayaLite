# LASER 原地更新:论文故事线("量化不冻结图")

> 目标:把 `LASER_UPDATE_EXPLORATION.md` 的工程成果升华为一篇独立研究论文的叙事。
> 轴线:**通用更新方法放到 LASER 上不行(逐个失效)→ 恰恰是 LASER 的结构让原地更新
> 比 raw 格式更顺 → "量化不是更新的障碍,而是更新的杠杆"**。
> 本文 = 故事底稿 + 失效分类学 + 主张/证据/缺口矩阵 + 审稿攻击预案 + 论文骨架。
> **裁决版**:codex 并行叙事评审(`LASER_UPDATE_PAPER_STORYLINE.md`,464 行,作附件)
> 已交叉合并,分歧裁决记录见 §10;细节(12 攻击全文、13 图各自证明什么、逐条成本估计)
> 以附件为准。

## 0. 定位与认领边界(先对齐红线)

- **这是"下篇研究论文"**:工业轨 AlayaLite 论文明确"不做 LASER 可更新化,留下篇研究
  论文"(story-v3.1 §6.5 已铺桥:PQ 漂移无罪 1.0008×、Clustered runbook 测码本开裂、
  HAKES 热切换 IVF 先例)。本故事按 **VLDB 研究轨**口径写(注:codex 附件按工业轨
  校准——简报笔误——其建议绝大部分对研究轨同样成立;研究轨免部署故事义务,
  但 novelty 文献核查义务更重)。
- **不可认领**:LASER 格式本身(AlayaLaser,SIGMOD 2026,自家已发表——需
  "relationship to prior work" 段);"first in-place graph update"(IP-DiskANN 占);
  "first RaBitQ in graph"(SymphonyQG 占);Yi 匿名在审期间不得提名(引 IP-DiskANN/
  FreshDiskANN 公开谱系)。
- **独有方格**:原地更新 × **边上量化打包**盘格式。已知的原地更新系统
  (FreshDiskANN→IP-DiskANN→OdinANN/PipeANN/DGAI)全部工作在 raw-vector DiskANN
  格式(盘上行 = 原始向量 + 明文邻居 id 表);量化图谱系(SymphonyQG/LASER/RaBitQ 系)
  全部静态。两条线的交点没人占——而且**没人占是有原因的**,这个"原因"就是本文的靶子。

## 1. 中心命题(thesis)

主命题(裁决采纳 codex 收窄版——机制主张,每个词都有实验背书,不越权比较):

> **LASER 把看似令图冻结的边上量化布局,反过来变成了局部更新原语:拓扑改一条边,
> 量化状态也只改这一条边;定宽空槽吸收新回链,行内打包码直接为驱逐和删除重连提供
> 零额外扫描 I/O 的距离信号。**

对 raw 格式原地系(IP-DiskANN 谱系)的定位句(全文最像 insight 的一句):

> **量化载荷是额外义务,也是额外信息**——同样的原地协议搬到 LASER 上必须多维护
> 每条边的量化码(义务,因端点局部性而闭合),但码常驻行内又反过来消除了驱逐与
> 删除重连的候选扫描 I/O(信息,raw 格式没有)。

激进版"量化不冻结图/比 raw 更适合更新"**维持降级**:漂移 v0 = 零结果(§11.2b,
温和 SIFT 漂移 + rerank 兜底下冻结 PQ 不裂;初报正向系掩码伪影,已撤)。主命题
锁定机制档;激进档解锁条件 = v1 强漂移(跨域/真时序/无 rerank 工况)出正结果。
方法论红线:masked-recall 的 pre→post 双差在人群构成变化的组上不可用,一律用
同终态静态参照口径(§11.2b 教训,写进论文评测方法节)。

三个支撑翻转(每个都是"通用视角的障碍 → 结构视角的杠杆"):

| # | 通用视角(为什么大家不碰) | 结构事实 | 翻转后的杠杆 |
|---|---|---|---|
| F1 | 量化码 = 全局码本产物,更新⇒漂移⇒重训练 | 边 (u→v) 码 = sign(rot(v)−rot(u)),FHT 数据无关,因子只依赖两端 raw 向量(就存在行头) | **更新影响半径 = 一条边**;patch 与整行重算逐字节等价(单测锁定),永无重训练 |
| F2 | FastScan 交织打包块不透明,动一码毁一块 | 打包是可逆置换(unpack→改槽→repack,往返测试锁定) | 外科手术式改一槽,块内其余 31 槽不动 |
| F3 | 定宽行无 append 余量,加边⇒全行重剪枝 | 定宽行字节先付 + 纯 id 算术寻址 | 空槽 = **免费更新余量**(零增量行字节;欠满建图不损静态质量,流式插入点 ef≥100 逼近同参数静态上界(多跑包络 ≤0.5pp,§14 修订)、低 ef 统计持平——"反超"版不可复现,禁用;§11.1b);行满时行内自带码 = **免费距离估计器**(evict 补丁 ≈ 1/9 成本逼平 full RobustPrune);追加即插入,免映射表 |
| F4(删除侧) | 删除修复要拉全邻域向量,量化行更没法修 | 死节点行里躺着它全邻域的码 = 现成的距离估计器 | **splice-from-corpse**:零额外 I/O 完成重连,consolidate 后零死边 |

一句话备选(按锋利度排序):
1. "Quantization does not freeze the graph — it is precisely the codebook-free, edge-local
   quantization that makes the graph *more* updatable than raw formats."
2. "The update radius of an edge-quantized graph is one edge."(机制版,适合 §设计开篇)
3. "Fixed-width rows are not rigidity but prepaid headroom; packed codes are not opacity
   but free distance estimators."(F3/F4 合并版,适合 abstract 第三句)

## 2. Intro 张力:假两难(narrative arc)

1. **读路径的现状**:盘上量化图(LASER/SymphonyQG 谱系)是单盘检索 SOTA——单跳 I/O、
   FastScan SIMD 批距离、行内自带 raw 向量免二跳 rerank。
2. **更新的现状**:整条可更新谱系(FreshDiskANN 的 fresh 层+merge → IP-DiskANN/
   OdinANN/PipeANN/DGAI 的原地)**全部**建立在 raw-vector 格式上。fresh 层付 merge
   全量重写 + recall 台阶(FreshDiskANN 97.91→95.4);原地系付 raw 格式的读路径代价。
3. **为什么没人做"量化图 + 更新"**:三个表面障碍(§1 表左列)。这不是懒惰,是合理
   直觉——如果量化真需要全局码本,冻结就是对的。
4. **假两难陈述**:今天想要 SOTA 读路径就得接受静态索引,想要更新就得退回 raw 格式。
   本文消解这个两难,且方向出人意料:量化格式不是勉强可更新,而是**更好更新**
   (headroom 逼近同参数上界、splice 免 I/O、无码本可漂移)。
5. (钩子)我们的通用方法尸检:把业界标准做法逐个搬上 LASER,每个都失效或低效,
   失效机理恰好指向三个结构事实——解法不是绕开结构,而是使用结构。

## 3. 通用方法失效分类学(G1–G7:機理 + 实证 + 解法锚点)

> 论文里这是 §2/§3 的骨架:每个 G 一小节,先给通用做法,再给它在 LASER 上的下场
> (有实验的贴数字,没实验的标注),最后一句话指向我们的杠杆。这个"尸检队列"
> 就是"我们的解法不是可选项而是必然"的论证。

| # | 通用做法 | 在 LASER 上的下场 | 机理 | 实证 | 我们的解法 |
|---|---|---|---|---|---|
| G1 | append-only(只追加新行,不动老行) | **new_recall = 0**,插入点完全不可达 | 图可达性靠反向边;不写老行 = 新点是孤岛 | E1 none 臂(§5) | 无解可绕:**必须 RMW 打包码区**——问题被逼到墙角,后续 G 全部由此展开 |
| G2 | DiskANN 原地协议直移植(反向边 = 邻居行 append 一个 id) | 每条反向边变成"整行重算":2303 读/插入、205 inserts/s,**9× I/O**;质量并不更好(recall 与廉价补丁重合,插入点甚至略低) | DiskANN 加边写 4 字节 id;LASER 加边 = 对 (v,u) 做边内量化并插入交织块——把打包块当不透明 blob 就只能全行重来 | E1 full 臂 vs evict 臂(§5) | F1+F2:单边重量化 + 可逆置换改槽,40 读/插入拿到同等质量 |
| G3 | 行满即全行重剪枝 / 加间接层扩容 | 重剪枝 = G2 的 9×;间接层毁掉纯 id 算术寻址(加映射表、断页对齐) | 定宽行没有 append 语义,通用格式思维只有"重写"或"分裂" | E1 成本表;E1b | F3:evict 用行内码免费估距选最差边替换;headroom 欠满建图预付余量,插入点 ef≥100 距同参数静态上界多跑包络 ≤0.5pp、低 ef 噪声内持平(§11.1b+§14 复核),远超无余量流式(evict 0.9506@ef60);"反超"表述禁用 |
| G4 | 全局码本系的更新观(PQ/IVF:更新⇒码本漂移⇒重训练/重建) | 若 LASER 用全局码本,同样死;但 LASER 没有码本 | 逐边相对码 + 数据无关 FHT,量化没有任何全局训练产物 | 漂移 v0 = **零结果**(§11.2b):温和簇漂移 + rerank 兜底下冻结 PQ 不裂;G4 的"通用失效"仅在无 rerank 兜底/强漂移工况成立 | F1;边界收紧:免码本是机制优势(永无重训义务),不是 SIFT 尺度的 recall 优势;LASER 不为格式付额外漂移代价(防守性) |
| G5 | fresh 层 + 周期 merge(FreshDiskANN 路线) | merge 在量化格式上 = 全量重打包;RAM + 双索引查询 + merge 风暴;而纯原地已达标 | fresh 层是"盘上行不可变"假设的补偿;假设一破,复杂度失去存在理由 | 纯原地 13.2–14.7k inserts/s + churn recall 平(§9);四个触发门未触发(§8.4);**fresh 基线实测缺位(Gap-2)** | 整个 §设计:原地协议本身 |
| G6 | 删除 = tombstone + 定期重建 | 无整形 churn 每轮 −0.5~−2pp 缓降且加速,100% 换血 −10.3pp;重建 87s/1M 且离线 | 墓碑 ballast:活节点出边指向死区,beam 浪费 | E3;consolidation 对照(§7.2:0.9748 vs 0.8887) | F4:splice-from-corpse 零额外 I/O 重连,4-5s/轮在线整形,零死边不变量 |
| G7 | 通用写路径调优(O_DIRECT、批末写回、io 调度) | 每补丁同步 DIO 更差(5.9k vs 7.4k);批末写回两种口味都输(134k/70k 页/s);buffered inode 锁 287k pwrites/s 顶 | 反向边 RMW 是 hub 偏斜的小写,**写次数才是敌人,不是写口味** | §9.1–9.2 三连判决 | 驻留 buffer pool 跨批合并(35.8→8.6 写/插入)+ inline 补丁(相位分离是 2× 代价)+ 分配器串行点手术;13.2–14.7k inserts/s |

注:G1/G2/G3/G6/G7 已有一手数字;G4 是主攻缺口(见 §6);G5 可用文献数字
(FreshDiskANN merge 2× 重写、recall 台阶)+ 成本核算替代实测,但审稿人可能要实测。

**三分法纪律(codex,采纳)**:失效要分三类写,避免稻草人——
① **硬失效**(G1:可达性条件不满足);② **可工作但没利用结构**(G2/G5/G6:协议
骨架可移植,但 raw-vector 数据面不维护边码,也拿不到行内码的免费距离信号——fresh
层"延后而非消灭边码写");③ **未实测不能下结论**(G5 的"原地全面优于 fresh"、
G4 的"免码本 recall 优势")。论文必须严格区分"我们没做"与"别人不行"。

## 4. 贡献点(C1–C5,证据锚点)

| # | 主张 | 证据 | 锚点 |
|---|---|---|---|
| C1 | **边上量化打包盘格式的原地更新协议**(该格式谱系此前全静态):插入 = 搜索捕获→α-prune→追加行→逐邻居单边重量化补丁;patch 与整行重算逐字节等价 | 单测 5/5 + 11/11;E1 evict≈full | EXPLORATION §3–§5 |
| C2 | **质量有界收敛**:100% 换血 recall@100 = 0.9865/0.9878,50 轮尾段 −0.005pp/轮≈平;流式 vs 静态全建差 0.1–0.5pp;headroom 插入点距同参数静态上界 ≤0.35pp 且欠满建图不损静态质量 | E1/E1b/§11.1/§9.4 长时域 | §5、§9.4 |
| C3 | **写引擎**:单盘 13.2–14.7k inserts/s(64/128T),物理写 8.6 页/插入;三个可迁移判决(写次数>写口味;相位分离是隐性 2×;glibc 堆伸缩是并行写路径的全局串行点) | §9.1–9.3 三连判决 + 消融 | §9 |
| C4 | **删除与整形**:懒删近免费(10% 时 ef≥200 与重建重合);splice-from-corpse 零额外 I/O 重连,churn +8.6pp,在线 4–5s/轮 | E2、§7.2 | §5、§7 |
| C5 | **失效模式的跨格式洞见**:量化格式改变退化机理——LASER churn 残差主体 = 插入时边质量欠账(ef_insert 100→200 单项 +0.92pp),而非 DiskANN 式入度年龄衰减(evict 补丁每轮 30 万次驱逐天然给老节点换血);gardening 在 LASER 上是配角且随时域复利(+0.12→+0.46pp) | §9.4 十臂消融 + DiskANN 修复线一手对照(fig13 解剖:同作者可引) | §9.4 |

C5 是差异化亮点:同一批作者手上有两种格式的第一手退化解剖,可以写出
"format shapes failure mode" 这种跨系统结论,别人没这个对照数据。
**裁决(codex A14 采纳)**:当前是单数据集、非受控对照的经验洞见——不进摘要、
不进贡献列表首位,放 discussion;受控对照实验(§6 Gap-8)做完才可升格。

## 5. 审稿人攻击预案(按杀伤力排序)

| # | 攻击 | 防御 / 需补 |
|---|---|---|
| A1 | "免码本优势没有实验:稳态 SIFT churn 下 PQ 也不漂移(你们自己测的 1.0008×)" | **v0 已跑,零结果**(§11.2b):静态参照腿显示温和 SIFT 簇漂移 + 全精度 rerank 下,冻结 PQ 无可检出码本特异退化(两系统干净交互项 −1.24 vs −1.12pp 不可区分);初报 −4.2pp 系掩码人群伪影,已撤。免码本回归**机制主张**;recall 优势兑现条件 = 跨域强漂移/rerank 受限/小 PQ 预算(v1,P1 级可选)。防守性正结论:LASER 老区流式近零损(g0 −0.06pp),OOD 惩罚与 raw+PQ 同量级 |
| A2 | "没有 fresh-layer/FreshDiskANN 基线实测" | Gap-2:最小实现或成本核算(merge = 全量重打包,量化格式上成本可解析算出)+ 文献数字;四触发门论证(§8.4) |
| A3 | "规模只有 1M,单盘,页缓存热" | Gap-3:deep10m/100m 复跑主表;冷缓存臂;NVMe 直连口径 |
| A4 | "更新期间的查询呢?并发读写干扰曲线缺失" | **已封口**(§15,24 格矩阵):全速插入(49.6k/s)下查询 p50 0.54ms/p99 3.42ms 共存,recall 罚 0.61pp;**池耦合净收益为正——0% 插入反而是最慢格(冷盘),更新流即预热流**;维护相位不挡查询(consolidate 36% seqlock 重试=微自旋,p99 1.55ms);唯一坏尾巴 insert+delete p99 16ms 如实报告并排 P3b |
| A5 | "崩溃一致性?原地覆写 = 掉电毁图" | P3 full-page redo WAL 设计 + 开销测量(工业线同款攻击,防御方案已有) |
| A6 | "headroom 反超是不是 R56 vs R64 的参数巧合" | **已跑 + max 复核**(§11.1/§11.1b):"反超"不可复现(ef60 双向均在噪声带);hr 插入点 ef≥100 距同参数上界多跑包络 ≤0.5pp、低 ef 统计持平。主张定稿"零字节成本逼近同参数上界";附带方法论红线:低 ef 单跑差 <0.5pp 是噪声,须外部独立进程 ≥3 跑取均(§14:进程内 --runs 不取均且共享完成序;根因=异步完成序,非并列) |
| A7 | "evict 的估计误差长期累积?" | **已封口**(§11.3):exact-evict 臂 10 轮 churn 零增益(±0.4pp 双向抖动);遥测 rank regret p50=0–1、相对误差 1.67→1.69% 全程平;argmax 分歧(43–53%)全发生在近平手场景 |
| A8 | "吞吐口径:ef_insert=200 时多少?"(质量主杠杆的代价) | 诚实报告质量/吞吐前沿(ins100 vs ins200 两点已有,补前沿曲线);冷池单 shot 预计减半 |
| A9 | "salami-slicing:与 AlayaLaser/工业轨论文什么关系" | Relationship-to-prior-work 段:格式归 AlayaLaser,本文只认领更新协议 + 写引擎 + 退化解剖;工业轨明确排除本课题 |
| A10 | "α-检验在 churn 下更差(−12.8pp),你们的 RobustPrune 变体是不是坏了" | 转化为发现:churn 场景要入度不要保守剪枝(E3 读数);终选配置已弃 alpha 臂 |
| A11 | "full 臂不公平:逐邻居随机读没享受你们的 buffer pool/批处理,9× 是实现税不是算法税"(codex) | **已封口**(§11.3):同池同并行下 exact 2.8×/full 4.9×,均无质量增益;论文口径改用 2.8–4.9×;意外加强:full 在 ins200 并行下 new_recall 崩 8pp(深池 α-过度剪枝 + 行粒度重写的并发敌意)——直移植"不只贵还更差" |
| A12 | "全部实验在 128D 免 PCA 免残差的最易路径上"(codex) | **已封口**(§17 D0 + §18 D1):96d 全主维 pd128(npp=2 多行页)与 GIST 960d 四臂(PCA 残差路径,8/12/16KiB 多扇区页)全链路实测;高维设计点 main_dim=512(残差 448 维承载 1.66% 方差,ef200 −0.40pp 换 25% 页/盘)——残差切分在高维是设计甜点,与 96d 全主维判决构成"按方差选档(≥98% 处切)"的完整叙事;property test 扫参数空间仍留 P2 加固项 |
| A13 | "无槽复用,文件只增不减,不能叫 sustained update"(codex) | P2 free-list + 墓碑持久化 + 零入度后复用;跑 10× 换血证明文件大小平台化;做不完则标题/摘要避开 space-bounded,collection compaction 列为必要组件 |
| A14 | "LASER vs DiskANN 失效模式不同的结论缺受控因果——两系统 R/剪枝/预算/缓存全不同"(codex) | C5 降级:不进摘要,放 discussion 作观察;要升格需同数据同 trace 受控对照(LASER 加 no-evict 臂 / DiskANN 加 nearest-only 回链臂,看交互项) |
| A15 | "度量口径不一:E1 用 recall@10、churn 用 recall@100,单 seed,挑 ef 点"(codex) | 统一 recall@10/100 双报 + QPS-recall Pareto + 插入点/老点/全体三组;≥3 seeds 置信区间;固定 QPS 与固定延迟双口径。**seed 误差棒已建立**(§16 干净口径:数据级洗牌 + 连续水位发布,随机 seed 间 r10 散布 reuse 0.05pp / append 0.09pp,含顺序插入臂保守 0.24pp;随机序一致优于顺序序;旗舰配置(garden 开)下 reuse 反超 append)。多跑协议:外部独立进程(§14) |
| A16 | "维护配方只在 SIFT128 上调过——高维(嵌入主场)churn 平台离静态重建多远?"(E3 实测自曝) | **实锤在案**(§19):GIST960/pd512 100% 换血 r10 距同人口静态门 −2.3~−3.3pp(SIFT ≤0.5pp),顺序/seed 因素已排除;回应线=E5 剂量前沿(ef_insert/garden ef/frac/headroom 扫描,量化"收复 pp × 吞吐成本"),叙事定位=维护深度是随维度上升的显式成本旋钮,论文须给前沿曲线而非单点配方;若剂量收不完则升级结构项(splice 主维估距噪声/退化图复利) |

## 6. 缺口 → 补实验优先级

| Gap | 实验 | 支撑 | 成本估计 | 优先级 |
|---|---|---|---|---|
| 1 | **分布漂移 churn**:建图用 80% 聚类,插入流来自 held-out 聚类(big-ann clustered runbook);对照臂 = 冻结码本的 PQ 系原地更新(DiskANN PQ 候选路径);测新簇查询 recall 曲线 | G4/F1(免码本的 recall 优势) | 中(runbook 现成,PQ 对照要搭) | **P0,主攻图** |
| 2 | fresh 层基线:最小 fresh(内存图)+ 周期 merge(全量重打包)实测,或解析成本核算 | G5 | 中 | P1 |
| 3 | deep10m(/100m)主表复跑 + 冷缓存臂 | A3 | 中(机器已有,数据已有) | P1 |
| 4 | ~~并发读写干扰曲线~~ **已补**(P3a d065076 + §15 矩阵;剩复用槽 mixed 扩展) | A4 | 已完成 | ✓ |
| 5 | headroom 公平对照:升级为二维 sweep(build_R∈{40..64} × degree_bound∈{56..80},同行字节预算),不止 R56 单点 | A6 | 低-中 | 单点对照**已完成**(§11.1,判决=反超不成立、主张降级);全 sweep 降 P2,随多数据集批次做 |
| 6 | 质量/吞吐前沿曲线(ef_insert 扫描 × garden 剂量) | A8 | 低(脚本现成) | P2 |
| 7 | WAL 设计 + 开销 | A5 | 高(P3) | P2(设计章 + 微基准即可;若摘要出现 production-ready 字样则升 P0) |
| 8 | **exact-evict 臂 + evict 误差遥测**(FastScan argmax vs 精确 argmax 命中率、rank regret、被逐边真实距离、按轮累积曲线);full 臂同池化公平化 | A7/A11 | 低-中(臂现成加一个) | **P1,便宜且封两个机制攻击** |
| 9 | 300%+ 换血 + 槽复用后文件大小平台化 | A13 | 中(依赖 P2 槽复用) | P1 |
| 10 | 口径统一:≥3 seeds、recall@10/@100 双报、三组分层(插入点/老点/全体)、非 2 次幂维度数据集(**96d+960d 双端已实证**——§17:96d 全主维 0.9883@ef100、npp=2 多行页;§18:GIST960 主臂 pd512 0.9740@ef200、8/12/16KiB 多扇区页全点火、pd512 churn P2 终态核查全过。LASER 主场=高维已被数据坐实:残差切分在 960d 是设计甜点(−0.40pp 换 −25% I/O),在 96d 是劣化(残差方差 10.3%);嵌入集 768/1536d 为 Phase C 收尾腿) | A12/A15 | 中(机时为主) | P1(随规模批次一起跑) |
| 11 | LASER/DiskANN 失效模式受控对照(同 trace,LASER 加 no-evict 臂、DiskANN 加 nearest-only 臂) | A14/C5 升格 | 中 | P2(不做则 C5 留 discussion) |
| 12 | 高维维护质量缺口(GIST960 −2.3~−3.3pp):**前沿已扫**(§20:ins400 +0.72pp/−35% 吞吐、garden 高维零效、ins600 探顶仍差 −1.38pp=剂量枯竭)⇒ **侦查收束(§22-25):侵蚀非复利**(fresh≈静态零衰减),五嫌全排除(估距误选/splice/驱逐量/garden[反而+0.75pp 保护]/补丁完整性),反修复定理(turnover 靶向 −5.4pp:hub 的机会主义积累边是承重结构,重写必须 hub 规避)⇒ 定性=**有界内在维护成本,且失效模式数据集依赖(§27 反转)**:GIST=老化侵蚀(五嫌排除+hub 反靶),真嵌入=插入欠账(fresh<old,机会主义积累反而净增益,ins400 直接对症,缺口收窄至 −1.2~−1.7pp);论文口径=前沿曲线+年龄谱法证(同一仪器诊断两种病)+有界衰减+hub-aware 红线;探针记档:incoming-route diversity(GIST 侧)、freshman boost(嵌入侧) | A16 | 中(诊断 ~1h;修复未知) | **P0(高维是主场;论文口径=显式前沿曲线+归因,非单点配方)** |

## 7. 论文骨架(图表即论证)

- **Title 候选**(裁决序):首选 codex 版 **"Quantization as an Update Primitive:
  In-Place Evolution of a Packed On-Disk Graph"**(锋利且不越权);次选
  "Quantization Does Not Freeze the Graph: ..."(更响但隐含全称比较,drift 实验
  落地后才敢用);保守垫底 "In-Place Updates for LASER's Edge-Quantized On-Disk
  Graph"(drift 不支持强优势时的退路)。
- **Abstract 六句**:① 量化盘上图 = 读路径 SOTA 但全谱系静态;可更新系统全部退守
  raw 格式(假两难)。② 共识认为量化冻结图;我们证明因果相反(命题句)。③ 机制:
  逐边免码本量化 ⇒ 更新半径一条边(patch 逐字节等价);定宽行空槽 = 预付余量,
  打包码 = 免费估计器(evict/splice)。④ 写引擎:驻留池 + inline 补丁,单盘
  13–15k inserts/s、8.6 页写/插入。⑤ 质量:100% 换血 0.987、50 轮有界收敛;
  headroom 零字节成本使插入点逼近同参数静态上界(≤0.35pp)。⑥ 跨格式洞见:量化格式改变退化机理(插入深度
  主杠杆 vs 入度老化)。
- **Fig 1(money shot)**:读路径 QPS@recall × 可更新性的二维格局图——raw 原地系
  (读慢/可更新)、量化静态系(读快/冻结)、fresh 层系(读中/merge 台阶),我们
  占右上角。
- **Fig 2**:行格式解剖 + 三个结构事实 + 单边补丁手术示意(unpack→requant→repack)。
- **Table 2**:G1–G7 失效分类学(通用做法 × 机理 × 下场数字)——"尸检表"。
- **Fig 3**:none/evict/alpha/full 四臂 recall 曲线 + I/O 柱(G1 零、G2 9× 不更好)。
- **Fig 4**:headroom(插入点 recall vs 同参数静态上界 vs 无余量流式;卖点=零字节成本收窄差距,非反超)。
- **Fig 5**:churn 长时域(无整形加速下坠 / +consolidation 拉平 / +garden / 终选
  50 轮平台)。
- **Fig 6**:写引擎瀑布(写次数 35.8→8.6;吞吐阶梯 1.8k→8k→13.2k→14.7k;
  三判决各一格)。
- **Fig 7**:失效模式对照(LASER 插入深度消融 vs DiskANN 入度老化解剖)。
- **Fig 8(Gap-1)**:分布漂移下免码本 vs 冻结码本 recall 曲线。
- **Table N**:相关工作方格(格式 raw/量化 × 更新 静态/fresh/原地)——空格即贡献。
- **评测按研究问题组织**(codex,采纳;不按 E0/E1 实验日志顺序照搬):RQ1 量化行能否
  局部正确修改 → RQ2 量化辅助 evict/splice 是否低成本保质 → RQ3 headroom 跨参数成立?
  → RQ4 驻留池超内存规模可扩展? → RQ5 churn 有界?残差来自出生边质量还是年龄?
  → RQ6 vs fresh 层/raw 原地/重建何时胜? → RQ7 免码本在漂移下有实测收益?
  完整 13 图 ×4 表清单(每张证明什么)见附件 §7.4。

## 8. 相关工作定位(一句话each;Yi 在审期间引公开谱系)

- FreshDiskANN:fresh 层 + merge,raw 格式;merge 2× 全量重写 + recall 台阶——我们免
  fresh 层且格式量化。
- IP-DiskANN(及 OdinANN/PipeANN/DGAI 原地谱系):原地更新,但盘上行 = raw 向量 +
  明文 id 表,加边是写 id 不是重量化;"原地更新量化打包行"仍空白(PipeANN 明确
  不复用 id;我们槽位复用 + 暗至发布)。
- SPFresh(SOSP'23):IVF posting list 原地(LIRE),非图、有码本(SPANN 系)。
- SymphonyQG(SIGMOD'25)/ AlayaLaser(SIGMOD'26):边上量化格式的定义者,全静态;
  本文 = 该格式的动态化,格式贡献归前作。
- HAKES:IVF 码本热切换先例——反衬"免码本则无需热切换"。
- CleANN(内存动态图)/ Starling(盘上布局优化,静态):不同层。
- OdinANN(FAST'26)/ PipeANN(OSDI'25)/ DGAI / MicroNN:工业轨调研已确认的
  raw 格式原地更新占位者——"宽口径盘图原地更新"空白已死,本文方格必须钉死在
  **量化打包行**上(此为我方红线情报,codex 附件未覆盖)。

**最近邻竞品是四个坐标不是一个**(codex,采纳):协议最近 = IP-DiskANN 谱系;
架构最近 = FreshDiskANN;格式最近 = SymphonyQG/AlayaLaser;思想最近 = SPFresh
(局部修复代替重建)。novelty = 三条坐标的交点,措辞定稿(codex 版,文献核查
CleANN/OasisANN 后启用):

> *To our knowledge, this is the first in-place update design for a fixed-row,
> edge-quantized on-disk graph that makes topology repair quantization-aware
> without a delta graph or global re-encoding, and reuses the packed edge codes
> themselves for eviction and deletion splice.*

红线重申:Yi 匿名在审期间正文与引用**均不得出现 Yi**(引 IP-DiskANN/FreshDiskANN
公开谱系);"first in-place"/"first dynamic quantized ANN" 均过宽不可写。

## 9. 一段话故事(电梯稿,中文版)

> 盘上量化图是当前单机向量检索读路径的最优解,但整个谱系是静态的;所有支持更新的
> 系统都退守 raw 向量格式,付出读路径代价——因为业界共识认为"量化 + 打包 + 定宽 =
> 冻结"。我们把业界标准的更新手法逐个搬上 LASER 做尸检:只追加不可达(recall 0)、
> DiskANN 协议直移植贵 9× 且不更好、写路径调优撞在"写次数"墙上、tombstone 放任则
> churn 加速下坠。每次失效的机理都指向同三个结构事实:逐边免码本量化、可逆交织打包、
> 定宽行 id 算术寻址。翻转使用这三个事实,更新半径缩到一条边:补丁与重建逐字节等价,
> 空槽是预付的更新余量(欠满建图不损静态质量,流式插入点逼近同参数静态上界),行内打包码是免费的
> 距离估计器(1/9 成本逼平完整重剪枝;死节点行自带重连装置)。配上驻留页池写引擎,
> 单盘 13–15k inserts/s、每插入 8.6 页写,100% 换血后 recall 0.987 并在 50 轮内有界
> 收敛。结论:量化不冻结图——恰恰是免码本的局部量化,让量化图比 raw 图更适合原地
> 更新。

## 10. 交叉裁决记录(我方稿 vs codex 附件)

**一致**(双方独立得出,置信度高):最危险缺口 = 分布漂移实验 + fresh 基线,不是
WAL;写作纪律 = 已证的写坚决、未证的标待验证("故事不是缩弱,而是不让真贡献被
过宽 claim 拖垮");投稿前最小实验包序:drift → fresh 基线 → 超内存规模/多 seed →
并发干扰曲面 → exact-evict/headroom sweep。

**采纳 codex**:① 主命题收窄为机制主张(§1),"免码本优势"降为待验证;② 失效
三分法(§3);③ 五条新攻击 A11–A15(full 公平性/最易路径/空间无界/因果受控/口径);
④ C5 不进摘要(§4);⑤ 标题首选 "Quantization as an Update Primitive";⑥ RQ 制
评测组织;⑦ "额外义务 + 额外信息"定位句。

**采纳我方(红线情报,codex 不可见)**:① 赛道 = 研究轨非工业轨;② Yi 匿名在审
禁提名;③ AlayaLaser 自家前作需 relationship 段防 salami-slicing;④ OdinANN/
PipeANN/MicroNN 已占 raw 原地空白 → 方格必须钉死"量化打包行";⑤ 工业线一手数据
(PQ 漂移无罪 1.0008×)作为 A1 攻击的弹药与诚实边界。

**遗留分歧**(无需现在裁决):headroom 二维 sweep 的优先级(codex 判 P0,我判
"单点堵 A6 先行、全 sweep 随批"),等机时约束明朗再定。

---

*生成于 2026-07-12,基于 feat/laser-update-explore 全部实验;codex 交叉评审合并版
(附件 `LASER_UPDATE_PAPER_STORYLINE.md`)。*
