# pg_db4ai 公开表格数据 Benchmark：实际结果与预测优化

**执行日期：2026-10-16。** 本报告来自实际 Docker 执行，不是预估结果。可复算摘要、原始结果记录和审计信息已提交到 [results/2026-10-16](results/2026-10-16)。

## 结论

pg_db4ai 的 XGBoost 在本次 18 个“数据集×划分”对照中，与同源码 Python XGBoost 的独立训练输出一致：回归值逐元素一致，二分类标签一致；模型导出再加载后的原始预测也逐元素一致。这验证的是这些任务上的训练/持久化/推理路径，不是所有参数和未来版本的正确性证明。

不能说插件“全面比 openGauss 快”。原生 GD/NGD 训练目前通常更慢，Gaussian 特征映射路径有明显开销；另一方面，所测 openGauss 6.0.0 的 XGBoost SQL 预测远慢于插件。后者包含 XGBoost 1.4.1 与 3.4.2、加载生命周期、wrapper 行为的差异，不应包装成纯数据库内核的加速比。

本轮移除了插件 gbtree 推理中逐行构造 DMatrix 的热点。最终同机 A/B 的六组任务提高 1.42–4.22 倍，模型字节和预测完全相同；此前独立同机复测为 1.23–2.56 倍。绝对延迟和加速幅度受 VM 环境影响，不能将最大值视为通用承诺。

## 1. 数据和测量范围

| 数据集 | 行数 | 特征数 | 任务 | 重复特征行 |
|---|---|---|---|---|
| Breast Cancer | 569 | 30 | binary | 0 |
| Banknote | 1372 | 4 | binary | 24 |
| Wine | 178 | 13 | multiclass | 0 |
| Digits | 1797 | 64 | multiclass | 0 |
| Diabetes | 442 | 10 | regression | 0 |
| Red Wine | 1599 | 11 | regression | 240 |
| Concrete | 1030 | 8 | regression | 34 |
| California | 20640 | 8 | regression | 0 |

数据来源及许可链接见 [方法说明](README.md#datasets-and-attribution)。Digits 是展开后的图像特征，Banknote 是图像提取的数值特征；不是八组企业业务表。所有输入是数值表格。KumoRFM 对应多表、时间和关系任务；本轮没有执行 KumoRFM、TabPFN 或复现其论文排行榜。

三组划分 seed=42/2024/7，每组约 80% 训练、20% 测试。相同特征向量被分组，避免重复行跨训练/测试泄漏；没有直接去掉重复行。StandardScaler 只在训练集拟合；回归目标也只用训练集统计标准化，报告质量指标时恢复原单位。没有使用测试集调参。

每组配置先预热一次，再计时三次。质量只计算每个划分最后一次模型的结果，不能把三次计时当成额外三个独立划分。最终 **744 条记录**包括 468 条数据库、252 条 Python 和 24 条均值/多数类控制记录，共 **264 个不同的“划分×方法”结果**（PG 78、openGauss 78、Python 84、控制 24）。三个 overlapping holdout 的均值/样本标准差不是显著性检验或总体置信区间。

## 2. 环境与公平性边界

| 项目 | 实际配置 |
|---|---|
| PostgreSQL | 18.6，Debian bookworm，pg_db4ai 0.2 + 本轮预测优化 |
| openGauss | 6.0.0 build aee4abd5；固定摘要的 enmotech Docker 镜像，不是移植时的 master 源码构建 |
| 插件/Python XGBoost | 3.4.2，同一份 vendored 源码和编译后的静态对象；Python 重新链接 C API shared library |
| openGauss XGBoost | 1.4.1，实际加载 C API 查询版本，并记录库 SHA-256 |
| 其他基线 | scikit-learn 1.7.2，LightGBM 4.6.0 |
| 每个测量容器 | 同一 CPU 绑核，1 CPU 配额、4 GiB、1 GiB shm，无外部网络 |
| 线程设置 | Python/BLAS/OpenMP=1；PG 查询并行关闭、JIT 关闭；openGauss query_dop=1 |
| 划分执行区块 | seed42：AMD EPYC 9V74；seed2024：EPYC 7763；seed7：EPYC 9V45。每个区块内基线同机串行；区块间不同 VM |

原始镜像缺少 libgomp.so.1，导致内置 XGBoost 不能加载；派生镜像只补齐发行版 libgomp1，不替换数据库或 XGBoost 二进制。数据库未显式统一的默认参数仍有差异；容器上限相同不等于实际峰值内存相同，本轮没有可靠峰值内存测量。

DB 训练计时包括读数/转换/训练/序列化/模型目录持久化；不包括构建、启动 Docker、启动 SQL 客户端、COPY 导入、预处理或传输数据。DB 预测是一个 SQL 内的“模型查找/加载 + 全部测试行预测 + SUM”，不包括把所有预测传回客户端。Python 则是驻留模型的向量化批量预测，load/serialize 单独记录；二者不是同一种 API，不能把其比值当成纯 SQL 开销。

GD/NGD 固定最多 200 轮，batch=128；GD 学习率 0.05，NGD 0.8；树模型固定 100 轮，depth=6，eta=0.1。全部参数及精确 SQL 均保存。sklearn 的 OLS/lbfgs/LinearSVC 不是原版 GD 的相同求解器。openGauss XGBoost 使用整份训练集作为 batch，避免旧 chunk-wise warm start，但其每轮评估等 wrapper 行为仍不同。

**迭代次数限制：** 原计时运行的 openGauss metadata 查询误用了 model_name，而实际列是 modelname；这不影响计时或预测输出，但 raw-results 中保留了 metadata_error。已修复脚本，并用完全相同的数据/划分 hash 做独立不计时的 78 模型审计：18 个 XGBoost 都为100轮，59个原生模型为200轮，Diabetes/seed7/linear_gd 为139轮提前收敛。这个审计不是原计时实例的元数据，不进行静默替换；该 GD 点不适合解释为固定200轮的吞吐差距。

## 3. 预测质量

### 3.1 分类：Accuracy（%，三划分均值 ± 样本标准差）

| 数据集 | PG logistic/OVA | openGauss logistic/OVA | sklearn logistic/OVA | PG XGBoost | Python XGBoost | LightGBM |
|---|---|---|---|---|---|---|
| Breast Cancer | 97.95 ± 2.03 | 98.25 ± 2.32 | 97.95 ± 2.82 | 95.91 ± 3.55 | 95.91 ± 3.55 | 95.61 ± 4.56 |
| Banknote | 98.44 ± 1.25 | 98.44 ± 1.25 | 98.68 ± 1.26 | 99.14 ± 0.77 | 99.14 ± 0.77 | 99.51 ± 0.56 |
| Wine | 93.52 ± 1.60 | 94.44 ± 2.78 | 97.22 ± 0.00 | 未暴露多分类 objective | 93.52 ± 3.21 | 93.52 ± 4.24 |
| Digits | 95.75 ± 1.36 | 95.75 ± 0.88 | 96.67 ± 1.27 | 未暴露多分类 objective | 96.86 ± 1.12 | 97.87 ± 0.64 |

二分类使用 logistic，多分类使用 One-vs-All logistic。这里不是在测试集上为每个数据集选最优算法。XGBoost 多分类基线仅在 Python 运行，插件本轮保持原有四种 objective，未临时扩充多分类树模型。

Banknote 的原生 logistic/NGD/SVM 在三组划分中的分类结果质量与 openGauss 一致。小样本 Wine 上每错一个测试样本就改变约2.78个百分点，因此不能放大小差距。Digits 的 LightGBM 在本固定配置下准确率更高；不能据此断言其在所有表格任务上优于其他算法。

**保留的不利结果：** Breast Cancer 的 Gaussian SVM 在 PG 和 openGauss 上都只有约56.43% Accuracy，等于多数类控制，尽管 linear SVM 接近97.95%。这组 gamma=0.5、128随机特征、lambda=0.5 的固定组合不合适，不能删除该结果，也不能据此认为所有非线性 SVM 都无效。NGD 在当前固定学习率下也并非总比 GD 好。

### 3.2 回归：R²（三划分均值 ± 样本标准差，越大越好）

| 数据集 | PG GD | openGauss GD | sklearn OLS | PG/Python XGBoost | openGauss XGBoost | LightGBM |
|---|---|---|---|---|---|---|
| Diabetes | 0.544 ± 0.044 | 0.543 ± 0.042 | 0.538 ± 0.048 | 0.361 ± 0.053 | 0.393 ± 0.063 | 0.411 ± 0.061 |
| Red Wine | 0.301 ± 0.043 | 0.301 ± 0.044 | 0.300 ± 0.043 | 0.302 ± 0.045 | 0.288 ± 0.061 | 0.321 ± 0.019 |
| Concrete | 0.608 ± 0.029 | 0.608 ± 0.031 | 0.610 ± 0.029 | 0.930 ± 0.016 | 0.931 ± 0.017 | 0.921 ± 0.012 |
| California | 0.602 ± 0.008 | 0.593 ± 0.029 | 0.609 ± 0.001 | 0.835 ± 0.003 | 0.834 ± 0.003 | 0.834 ± 0.002 |

Concrete、California 的树模型明显强于线性模型；Diabetes 的固定100轮树模型反而弱于线性方法；Red Wine 的模型整体解释能力有限。不能把“树通常 work”当成每个小表上的保证。完整 RMSE、MAE、F1、balanced accuracy、AUC 在 summary.csv 中；本轮没有提取 openGauss 原生分类模型的同口径原始概率，因此不伪造其 AUC。

PG XGBoost 的平均 RMSE：Concrete 4.349 MPa，California 0.4705 个十万美元（约47,052美元），Diabetes 60.974，Red Wine 0.6759。California 使用随机地区记录划分，不是跨地区/未来时间泛化测试。

## 4. 数据库性能

以下绝对耗时为每个 seed 内三次重复取中位数，再跨 seed 取中位数。**速度比则先在同机同 seed 配对计算，再取中位数**，所以不一定等于表中两个跨主机中位数直接相除。极短查询接近客户端计时噪声，不对十分之一毫秒差距作强结论。

### 4.1 XGBoost：训练与整张测试集 SQL 预测（ms）

| 数据集 | PG训练 | OG训练 | PG预测 | OG预测 | 预测 OG/PG 配对比 |
|---|---|---|---|---|---|
| Breast Cancer | 68.47 | 66.32 | 1.70 | 259.29 | 152.7× |
| Banknote | 18.82 | 30.97 | 2.62 | 860.12 | 338.4× |
| Diabetes | 50.65 | 91.16 | 2.04 | 762.43 | 374.1× |
| Red Wine | 70.78 | 129.78 | 3.62 | 3027.95 | 940.9× |
| Concrete | 73.80 | 125.65 | 2.93 | 2242.97 | 765.3× |
| California | 197.49 | 414.90 | 30.98 | 57223.69 | 1914.7× |

这说明所测 openGauss 6.0.0 的逐行 XGBoost 集成存在巨大预测开销，**不说明 PostgreSQL 内核或 XGBoost 3.4.2 算法本身普遍快150～1900倍**。此前审阅的原版 xgboost_predict 路径每行创建并反序列化 Booster，再创建 DMatrix、预测和释放；这与现象一致，但旧库版本、编译和 wrapper 也都是混杂因素。插件在本版本中按函数调用状态缓存已加载模型，并消除 gbtree 每行 DMatrix 构造。

XGBoost 训练并非全面领先：Breast Cancer 两者接近、插件略慢；其他五组的配对训练比约1.65～2.16倍。不能把预测倍数套到训练上。

### 4.2 原生算法训练：插件目前仍有明显开销（ms）

| 数据集 | 配置 | PG训练 | OG训练 | PG/OG 配对比（>1为PG慢） |
|---|---|---|---|---|
| Breast Cancer | logistic_gd | 19.13 | 7.56 | 2.54× |
| Banknote | logistic_gd | 17.65 | 8.13 | 2.51× |
| Wine | ova_logistic | 6.84 | 4.70 | 1.46× |
| Digits | ova_logistic | 483.72 | 400.63 | 1.21× |
| Diabetes | linear_gd | 7.42 | 1.99 | 3.57× |
| Red Wine | linear_gd | 20.25 | 8.76 | 3.12× |
| Concrete | linear_gd | 11.47 | 3.39 | 3.38× |
| California | linear_gd | 326.12 | 59.77 | 5.65× |
| Banknote | svm_gaussian | 1411.39 | 47.31 | 29.83× |
| Breast Cancer | svm_gaussian | 1208.30 | 24.39 | 49.54× |

原生线性/logistic/OVA 的训练约1.2～5.7倍开销，Gaussian 路径约30～50倍；这属于需要继续剖析的真实工程差距。仅由端到端时间还不能严格拆分为 tuplestore、类型转换、shuffle 或数值循环各自占比。不能把未经 profiling 的原因写成已证实事实。

原生线性/logistic 的预测通常是亚毫秒到约1ms级，两边相近；Gaussian 的预测也有一定差距。完整所有配置表保留在 per-split.csv/summary.csv，而非只保留有利任务。

### 4.3 Python 性能参照：计时边界不同（ms）

| 数据集 | Python 方法 | fit | 全部测试行批量 predict | load | serialize |
|---|---|---|---|---|---|
| Banknote | XGBoost | 19.151 | 0.342 | 0.673 | 0.679 |
| Banknote | LightGBM | 35.068 | 1.194 | 1.448 | 3.715 |
| Concrete | XGBoost | 69.264 | 0.420 | 0.923 | 0.850 |
| Concrete | LightGBM | 24.238 | 1.163 | 0.988 | 2.256 |
| California | XGBoost | 181.543 | 4.321 | 1.047 | 0.924 |
| California | LightGBM | 149.445 | 12.892 | 1.416 | 3.311 |
| California | sklearn OLS | 2.552 | 0.055 | 0.030 | 0.042 |

Python XGBoost fit 包含构建训练 DMatrix，predict 是已加载模型的向量化测试批次。Python serialize 为内存序列化，不包括持久化表提交。因此 California 的 PG 30.98ms 与 Python 4.321ms 不能直接解释成“数据库额外开销七倍”。它们提示仍有批量推理接口的工程机会。不同算法必须连同第3节的预测质量一起看，不能仅按耗时排名。

## 5. 本轮修复与同机 A/B

修改 src/xgboost.c：gbtree 复用 PG 内存上下文管理的 float 缓冲区及 XGBoost proxy，调用 XGBoosterPredictFromDense；不再逐行创建/销毁 DMatrix。保留 dart/gblinear 回退，异常清理与 float32 范围检查。训练过程与模型格式未修改。

主 A/B：同一 VM/CPU，before 为 bde6ee99fc14，after 为本轮实现；两边以完全相同数据训练，并先断言 UBJSON 字节一致、全部预测一致。每次使用新 backend，2次预热，11次配对重复，随机交替 old/new 先后次序。

| 数据集 | 测试行 | before ms | after ms | 主A/B加速 | 此前独立A/B加速 | 最大预测差 |
|---|---|---|---|---|---|---|
| Banknote | 279 | 12.475 | 5.275 | 2.36× | 1.75× | 0.0 |
| Breast Cancer | 114 | 7.213 | 4.419 | 1.63× | 1.36× | 0.0 |
| California | 4128 | 142.951 | 33.864 | 4.22× | 2.56× | 0.0 |
| Concrete | 206 | 11.501 | 6.136 | 1.87× | 1.47× | 0.0 |
| Diabetes | 89 | 7.105 | 4.996 | 1.42× | 1.23× | 0.0 |
| Red Wine | 320 | 15.178 | 6.849 | 2.22× | 1.65× | 0.0 |

主A/B和此前复测不混合成一个更好看的数。前者在 EPYC 9V74 上，早期复测另一个VM；均为内部同机成对。它们说明优化有一致方向的收益，但倍数依赖硬件/VM状态。源码测试同时覆盖 changing-row 输入、JSON/UBJSON 一致、gblinear/dart fallback 及异常后的恢复。

还修复了评测工程问题：镜像缺少 OpenMP runtime、openGauss 不接受 tolerance=0、COPY 0600 导出文件的归属读取、临时启动 server 的 readiness race、modelname 元数据列名。前几轮失败记录保留在 Actions/诊断分支；它们没有被冒充为算法质量差或速度结果。

## 6. 验证与可审计证据

主测运行：GitHub Actions run **38940991413**，测量源码 **95d0078efae9eff4ae5b6b86fdf6d464cb5730ea**。三个 seed 的训练/预测均成功，744条记录零算法执行失败。所有156个 DB“方法×划分”的原始预测由独立公式复算 Accuracy/balanced accuracy/F1、RMSE/MAE/R²，无差异；这项独立审计不包括AUC。18组同源码XGBoost训练对照和导出/加载 raw prediction 对照均零差异。检查了旧OG missing=-1 哨兵与本数据 float32 特征，没有精确等于-1的冲突值。

同一份 XGBoost C 修改已通过 CI run **38936740042**：PG16/17/18 with XGBoost、PG18 without XGBoost 的构建/安装/集成/动态依赖检查全部成功。后续主要是 benchmark 工具、审计和报告变化，不应把未改 C 的报告提交描述成重新训练了全部模型。

补充迭代审计 run **38941694021**；openGauss XGBoost 版本审计 run **38938685003**。结果摘要中包含原始 artifact ID、SHA-256、CPU、版本、源commit、数据和划分hash。完整表格数据、SQL、NPZ、模型输出、日志和镜像描述可在原始 artifact 与随附证据包检查。

**证据边界：** 本轮是8组小/中型数值数据、固定参数、单CPU、单连接/串行查询测试；不是 production-ready、峰值内存、并发、分布式、超大表、类别特征工程、relational model 或 SOTA 泛化能力证明。原生算法仍不能宣称原版逐位等价。增加参数搜索必须使用独立 validation split，不能根据这次 holdout 反复挑参数后再把相同holdout当作新评测。
