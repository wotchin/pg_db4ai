# DB4AI 0.3：功能核查、模型 DDL 与 CorgiPile 优化实测

执行日期：2026-10-16。测量源码：`e3da873258b0942d73b0166cf836599ec4895aad`。

## 结论

本轮没有运行 openGauss、sklearn、Python XGBoost/LightGBM 或旧插件作为基线。沿用已冻结的三组划分、预处理、预测输出和历史耗时；只编译、执行当前插件及当前代码的功能开关消融。

完成 122 组当前版本配置（60 batch、60 CorgiPile、2 个 Gaussian 特征缓存关闭对照），另有 9 组标签排序压力测试。默认 batch 路径的 60 组预测全部与冻结的旧插件预测逐元素一致。独立复算全部 131 组预测指标通过，未将三次计时重复当成独立质量样本。

按已冻结的 openGauss 耗时数值门槛比较，20 组原生算法配置的三划分中位数均低于该门槛。不过历史测试使用不同机器/时段，这不是同机证明。更强的当前同机因果证据是特征缓存消融：两组 Gaussian SVM 在模型内容不变时分别减少约28.53倍和39.69倍耗时。

CorgiPile 的目的不只是减少每轮耗时，还要改善有序数据上的训练混合。它在本次小表上并非普遍更快：60 个划分/配置的 CorgiPile/batch 耗时比中位数为1.285。标签排序压力测试在较少轮次上提高准确率，但20轮的最终准确率没有超过batch，因此保留为显式选项而非强制默认。

## 1. 功能范围重新核查

RNG 序列、KMeans 采样、随机 PCA 初始化/正交化、OVA 独立 NGD 状态是接受的实现选择，不再列为功能缺失。原版11个注册算法已有执行路径。

**ABO 不属于 DB4AI 迁移范围。** 不移植其利用 Bayes 进行优化器训练、计划选择和私有范围/边缘化的集成；保留 DB4AI 自身的 Bayes 算法。独立 snapshots 数据版本功能继续排除。

本轮补入实际缺少的用户能力：

| 能力 | 0.3 实现 |
|---|---|
| 模型命令语言 | 插件内 `execute_ddl` 和不执行训练的 `parse_ddl`，支持 CREATE/DROP/EXPLAIN MODEL 的主要命令形态 |
| KMeans 诊断 | 各簇坐标、行数、距离之和/最小/最大/均值/总体标准差；空簇统计为null |
| PCA 诊断 | 主成分坐标、所选特征值、所选成分内部占比、随机求解器估计量 |
| 小数训练时长 | `max_seconds` 支持小数，按算法迭代边界检查 |
| XGBoost 训练线程 | 可选 `WITH_OPENMP=1`，`nthread=1..64`；默认仍无OpenMP依赖、单线程 |
| 编译能力查询 | `db4ai.build_info()` 明确输出是否启用XGBoost/OpenMP以及原生grammar不支持 |
| 升级 | 0.2→0.3；已有本扩展模型不改写，仍可加载 |

PCA 统计中的比例是**所选成分内部的比例**，不冒充总解释方差。OpenMP 仅在上游 XGBoost 内部工作，不用于跨线程调用 PG API；逐行预测仍限制为单线程。完整范围见 `docs/FUNCTIONAL_COVERAGE.md`。

## 2. CREATE MODEL 的真实支持边界

原生 PostgreSQL 16–18 的内建 raw parser 会先拒绝未知 grammar，ProcessUtility_hook 无法接管一个尚未成功解析的 CREATE MODEL。可核查 [PG parser.c](https://github.com/postgres/postgres/blob/REL_18_STABLE/src/backend/parser/parser.c)。本版本不修改 PG 内核，也没有伪装成注册了顶层 SQL grammar。

已实现的入口是：

```sql
SELECT db4ai.execute_ddl($ddl$
 CREATE MODEL "房价模型" USING linear_regression
 FEATURES area, age*age TARGET price
 FROM public.houses WHERE area>0
 WITH optimizer='ngd', max_iterations=100, batch_size=64,
      shuffle='corgipile', shuffle_block_rows=128, shuffle_window_blocks=8
$ddl$);
SELECT db4ai.execute_ddl('EXPLAIN MODEL "房价模型"');
SELECT db4ai.execute_ddl('DROP MODEL IF EXISTS "房价模型"');
```

这是在插件里真正解析执行模型语言，但必须经过显式函数入口。`db4ai.parse_ddl(text)` 只返回结构化请求和生成的训练 SELECT。`db4ai.build_info()` 返回 `native_create_model_grammar:false`。直接发送裸 `CREATE MODEL ...` 仍不支持；需要原生裸语句体验时，必须另选核心grammar改动或客户端适配。

实现复用 PG lexer 处理注释、引号、字符串、括号和子查询，不是用正则截取SQL。支持literal/DEFAULT参数，参数名和适用算法校验、单一TARGET校验、调用者权限/RLS和训练函数EXECUTE检查。原版全部EXPLAIN格式选项、U&转义模型标识符和PREDICT BY表达式grammar未复刻；typed predict函数提供预测功能。详见 `docs/MODEL_DDL.md`。

## 3. 从论文吸收并落地的部分

论文：Xu et al., *Stochastic gradient descent without full data shuffle: with applications to in-database machine learning and deep learning systems*, VLDB Journal (2024), [DOI 10.1007/s00778-024-00845-0](https://doi.org/10.1007/s00778-024-00845-0)。其关键是块级随机选择/顺序与缓冲区内跨块行混排，而不仅是整批换序。

当前训练路径：

```
调用者SELECT（一次）
  → 原始tuple缓存与类型/标签处理
  → 连续数值重扫存储（超预算写入PG临时文件）
  → 固定特征映射预计算缓存
  → 随机逻辑块顺序
  → 多块窗口内行级随机排列
  → 固定大小minibatch
  → 梯度更新
```

新增 `shuffle:"corgipile"`；`shuffle_block_rows`、`shuffle_window_blocks` 与 `batch_size` 独立。每次完整数据遍历不漏不重，最后的短块/短窗口保留。CorgiPile 默认首轮也混排；旧batch/row默认首轮顺序保持不变。全部数据放得进窗口时才是全局均匀行排列，其余情况不是全表均匀洗牌。

存储块是**已授权查询结果的逻辑块**，不是直接随机访问heap page，因此仍适用于join/表达式，保持MVCC与权限边界。磁盘路径读取确切请求范围，避免随机小块被放大成整个较大的顺序缓存块。逻辑I/O计数不是物理设备流量。

论文的异步生产者/消费者双缓冲未实现；当前为同步块读取与两级混排。没有声称已复刻完整论文系统、物理页采样、GPU或多进程深度学习支持，也不直接搬用论文的收敛定理与加速数值。现有任意tuple SQL CustomScan `shuffle_scan` 继续保持原入口，训练CorgiPile是独立内部运行时。

## 4. 本轮性能优化的实际来源

两项不改变默认batch训练语义的优化：

**连续数值重扫。** 规范化后不再每轮用TupleTableSlot/Datum取出相同数值，而是直接消费连续double行/批。超过预算通过PG管理的BufFile溢写，事务异常清理仍由PG负责。

**固定特征只计算一次。** Gaussian/Polynomial随机映射固定后与当前迭代权重无关，可以先准备 `phi(x)`，更新和loss重扫复用。缓存也可溢写。通过 `kernel_cache:false` 可以在当前代码内关闭此优化做消融，而不是运行旧版本。

还修复了优化时发现的短批次边界错误：被洗牌到epoch中间的短unit仍须和后续行拼成原来的minibatch，不能提前执行一次小batch梯度更新。独立33行标量数学oracle验证batch/row/CorgiPile三种路径；真实数据60组默认输出与保存的旧输出完全一致。

## 5. 当前实测与冻结数值门槛

当前：Docker PostgreSQL18.6、native-only扩展、EPYC9V45 VM、1绑核CPU、4GiB、禁止网络、JIT和查询并行关闭。沿用三组冻结输入/顺序/预处理；每组1次预热+3次计时。以下为每划分计时中位数再跨3划分取中位数，单位ms。

**旧插件和openGauss列仅从历史结果文件读取，没有重测。历史硬件和构建配置不同，下面是数值目标对照，而非同机因果加速。**

| 数据集/算法 | 旧插件冻结值 | openGauss冻结值 | 当前插件batch |
|---|---:|---:|---:|
| Breast Cancer / Logistic GD | 19.128 | 7.561 | 6.713 |
| Banknote / Logistic GD | 17.653 | 8.128 | 6.383 |
| Wine / OVA Logistic | 6.842 | 4.704 | 3.528 |
| Digits / OVA Logistic | 483.719 | 400.629 | 270.391 |
| Digits / OVA SVM | 382.579 | 258.678 | 134.349 |
| Diabetes / Linear GD | 7.423 | 1.993 | 1.516 |
| Red Wine / Linear GD | 20.250 | 8.760 | 4.804 |
| Concrete / Linear GD | 11.472 | 3.393 | 2.713 |
| California / Linear GD | 326.125 | 59.770 | 51.790 |
| Breast Cancer / Gaussian SVM | 1208.304 | 24.389 | 20.444 |
| Banknote / Gaussian SVM | 1411.391 | 47.308 | 35.780 |

完整20组在 `historical-thresholds.csv`，均低于冻结openGauss数值。不能把这解释成所有机器上都领先；约1–3ms查询尤其容易受噪声影响。历史Diabetes/seed7/GD补充审计为139轮提前收敛，本次当前配置实际均为200轮，因此该历史点不能当作固定200轮吞吐基线。

### 当前同机缓存消融（seed42，ms）

| 数据集 | 关闭kernel_cache | 开启kernel_cache | 两者比值 | 训练更新/loss路径映射行数 |
|---|---:|---:|---:|---|
| Breast Cancer / Gaussian SVM | 805.276 | 20.290 | 39.69× | 182000 → 455 |
| Banknote / Gaussian SVM | 991.009 | 34.735 | 28.53× | 437200 → 1093 |

两边均为当前版本、相同数据/顺序/参数，模型MD5相同，另有直接模型bytea相等回归测试。200轮中，每轮更新与loss各计算一次映射，热路径由约400N次下降到N次。这是消除重复工作的收益，**不能归功于CorgiPile混排**。

Gaussian SVM的固定参数在Breast Cancer上仍然表现不佳：性能提升没有伪装成准确率提升，默认batch的预测质量与冻结结果不变。本轮没有用测试集重新调参。

## 6. CorgiPile本身的收益与代价

同一当前版本、8个小型数值数据集、60组划分/配置，CorgiPile/batch耗时比的中位数为 **1.285**，即这个汇总下CorgiPile多耗时约28.5%。这批训练输入本来已经冻结为随机顺序，进一步行混排有额外CPU/拷贝成本，不能指望普遍提高吞吐。具体归因仍以计数/阶段记录为限。

另设有序输入压力实验：只用已冻结的Banknote训练行，复制16次形成17488行后按label排序，原279行holdout不变，缓存预算256KiB。固定1/5/20轮分别独立训练，无外部基线，无新划分或holdout调参。这不是新收集的数据集，也不是一个模型的连续checkpoint轨迹。

| 轮次 | 当前batch Accuracy | CorgiPile Accuracy | batch训练ms | CorgiPile训练ms |
|---|---:|---:|---:|---:|
| 1 | 85.66% | 89.96% | 13.490 | 12.982 |
| 5 | 94.98% | 96.42% | 14.650 | 19.727 |
| 20 | 97.49% | 97.13% | 21.900 | 25.852 |

在该压力配置的较低轮数预算下，CorgiPile更早得到较好的测试结果；20轮未胜出。训练loss分别从batch的0.332071/0.145126/0.067245下降到CorgiPile的0.321797/0.142633/0.066988，但loss更低不保证holdout准确率更高。需要把统计效率与每轮执行成本分开看，不能只呈现有利轮次。

## 7. 验证、源码和复现

- 当前测量 run：38953253518，源码e3da873258b0942d73b0166cf836599ec4895aad。
- 完整CI run：38953289853；PG16/17/18 XGBoost无OpenMP、PG18 native-only、PG18 XGBoost+OpenMP共5组全部通过。
- 本地PG18.0完整OpenMP套件通过（含实际nthread=2训练）；native-only完整UBSan套件通过。
- 122组当前配置+9组压力结果全部独立复算；60组旧预测只从文件读取并逐行比较，全部最大差值0。
- 默认C/C11，直接依赖PG运行时；没有数据库无关层、没有ABO集成、没有新的外部数值库。
- 原始artifact **12070180256**，SHA-256 **a7957b06b0681bd6274172ffeb63561b439eaf699642b63152c69d16b8c3742f**。

`benchmarks/native/summarize.py` 只读取结果和冻结输入，重新核算质量/计时汇总/一致性，不运行估计器。已提交摘要保留全部不利结果、原始记录、三次计时、各阶段指标和来源。原始证据包还包含全部SQL、预测CSV、模型元数据、容器信息和执行源码。

没有对异步双缓冲、物理设备IO、峰值内存、并发/分布式、生产ready或表格任务SOTA作结论。0.3实现和功能边界分别见 `docs/CORGIPILE.md`、`docs/MODEL_DDL.md`、`docs/FUNCTIONAL_COVERAGE.md`。
