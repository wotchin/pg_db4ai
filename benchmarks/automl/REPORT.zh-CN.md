# 0.5：轻量 AutoML 与可复用预测 Pipeline

执行日期：2026-10-23。基于已合并 main `a131fbebc2c9c70916033073fa3f084497510a35`；测试/测量运行代码发布为 `94a86935c8fd8783e2e64be490e9f3a5ef706eb9`。后续报告和审计提交不改变运行代码。

## 1. 本轮交付

新增 `db4ai.automl(name, task, source_sql, options)` 和命名原始字段 JSON 预测 `db4ai.predict(name, raw_features)`。代码仍是 PG 专用 C11，复用现有 SIMD、训练数据重扫、类型标签、模型存储及内置 XGBoost，没有引入 Python 服务、外部 AutoML 框架、消息队列或第二套模型表。

自动化不只是遍历参数：源 SELECT 只执行一次；按任务划分训练/验证/最终测试；只在训练集拟合填补、缩放、类别编码；小样本筛选少量配置后，最多扩大两名候选；验证决定选择和是否发布；最终测试只作独立评估。完整变换和模型作为同一个版本化 artifact 保存，预测直接接受原始字段。

默认分类候选是 logistic/OVA-logistic 与 XGBoost，回归候选是 linear 与 XGBoost。每家两个固定的正则/学习率/树深配置，pilot 32轮/24棵树，再最多两次200轮/160棵树训练；原生算法可能提前收敛。通常最多6次fit，Gaussian NB显式开启后最多7次。`max_trials`是上限，不承诺用完所有时间或进行无限随机搜索。多分类 XGBoost `multi:softprob` 已进入 AutoML 的类型化 pipeline。

这是有预算的顺序候选集搜索，不是完整通用AutoML、元学习、贝叶斯超参优化或自动关系特征搜索。没有假装把“增加算法入口”作为本轮价值。

## 2. 预算与正确性约束

`time_budget_seconds` 从函数入口计时，涵盖取数、准备、搜索和持久化。更早的调用者 statement_timeout 不会被延长。80%用于准备/搜索，20%预留最终评估/发布；每个trial有可恢复的软截止时间。用户取消、外层超时、权限错误和恢复冲突不会被当成普通试验失败吞掉。源查询/最终化硬截止时整体报错并回滚，不发布半模型。

这些中断仍是协作式的，上游C++或长SQL算子可能延迟处理，不作硬实时保证。`memory_mb`为缓冲/准入限制，**不是整个backend的RSS硬上限**；单线程、输入行数/字节/编码量、模型大小与PG临时文件配额共同约束。超大输入明确报错，不偷偷训练前N行并称为随机抽样。

默认按不含目标值的特征哈希分组，相同特征向量不跨分区；支持实体组和时间顺序划分。比例约60/20/20，按组数而非保证行数；默认不分层。每类在三份数据中都要出现，否则拒绝，避免事后看分数重抽。时间划分不自动保证用户特征无未来泄漏。

数值NULL以训练均值填补，默认增加missing indicator；类别只保留有上限的训练词表，其他/未知值和NULL有独立桶。词表是训练首次出现顺序，不是频率最优选择。不提供类别target encoding。最终模型不在验证/测试或全输入上重训；pilot胜出可能使用更少训练行，报告明确列出selected_training_rows。

发布必须优于训练导出的多数类/经验先验或均值基准。`ready`是验证门槛，不等于业务验收和统计显著性；最终测试可能变差。`no_validation_improvement`和`no_completed_candidate`不创建模型。

## 3. 实测方法

当前版本、Linux x86-64 AMD EPYC9V74、PostgreSQL18.0、GCC14.2、XGBoost开启/OpenMP关闭、SIMD AVX2。postmaster及新backend固定CPU0，算法单线程，查询并行和JIT关闭。没有声称施加4GiB容器硬限制或测得峰值RSS。

沿用8个既有公开数据集、seed42/2024/7的24个冻结outer split。根据已保存缩放参数逆变换恢复原始数值（容许浮点roundoff）；只把outer TRAIN交给AutoML，再由其内部划分60/20/20。outer TEST不参与搜索/发布，也没有把它用于参数调整。

每任务预算2秒，一次预热、三次计时。质量只统计最后一次模型在outer TEST的结果，不把计时重复当作独立划分。计时来自服务端whole-call单调时钟，包含源读取、预处理、搜索、内部最终测试、序列化和目录写入；不包含psql启动/COPY导入。早期亲和性切换的探索运行单独保留，不混入最终耗时。

没有执行旧插件、openGauss、sklearn估计器、LightGBM或外部AutoML基线。当前模型只使用outer TRAIN内部的训练分区，而不是旧baseline的全部outer TRAIN，不能把耗时或质量直接包装成同训练条件的竞争胜利。

## 4. 完成的效果与成本

24组全部返回ready；72次正式计时的最小/最大总耗时为41.592/1255.321ms，全部低于2秒。下表质量为三划分均值，耗时为每划分三次中位数再取中位数。完整样本标准差保留在summary.csv。

| 数据集 | 任务/指标 | 外部测试质量 | 总耗时ms | 实际选中的家族 |
|---|---|---:|---:|---|
| Banknote | Accuracy | 98.79% | 83.823 | XGBoost |
| Breast Cancer | Accuracy | 96.49% | 149.708 | Logistic / XGBoost |
| Wine | Accuracy | 93.52% | 48.066 | OVA Logistic / XGBoost multiclass |
| Digits | Accuracy | 95.75% | 1054.301 | OVA Logistic / XGBoost multiclass |
| California | R² | 0.8319 | 986.803 | XGBoost |
| Concrete | R² | 0.8966 | 213.009 | XGBoost |
| Diabetes | R² | 0.4675 | 58.184 | Linear / XGBoost |
| Red Wine | R² | 0.3260 | 187.693 | Linear / XGBoost |

AutoML没有统一选树，也没有在外部测试集上选赢家。例如Diabetes seed42选中了验证集占优的树，outer TEST的R²仅0.3478；不能因为看到这个不利结果就改选别的模型。小样本单次holdout选模仍有方差，这不是泛化最优性或超过专业AutoML的证明。

### 紧预算补充检查

同一当前代码、8数据集的seed42、一遍预热和至多一次正式计时：50ms预算6/8任务返回合格模型，California/Digits返回no_completed_candidate；250ms预算7/8返回，Digits仍无完成候选。无完成候选的调用在预热阶段即保留报告，未启动正式重复；这些完成率是补充可用性观察，不是稳定服务SLO或三划分精度曲线。

50ms失败调用分别在44.340ms和37.541ms返回；250ms的Digits在147.306ms返回。候选局部预算可以提前终止全部尝试，而不是消耗整个总预算。源pg_sleep/更早父超时另有集成测试，验证明确报错、无发布以及后续调用可用。保留所有不利结果，未把失败填为0ms或满分。

## 5. 测试与独立审计

本地PG18 WITH_XGBOOST=1完整集成通过；native-only完整UBSan、halt-on-error通过。886项独立标准库预处理/原生预测公式核对，覆盖SIMD off/portable/auto、新连接与备份恢复。已有算法和数值套件继续运行。测试包括混合特征、NULL/常量/未知值、超大numeric标签精度、源执行一次、三种split、预算/GUC/计时器恢复、权限/RLS、回滚、损坏artifact、升级和恢复。

只改变最终test标签，选中的model bytea保持相同；训练词表不含test-only类别，训练均值由独立公式核对。XGBoost多分类和非线性任务均有实际测试。

`audit.py`只读结果，不运行数据库或训练器：复算24组共16598行outer prediction指标，检查96份含预热的选择/预算报告，并对1526行保存的原生模型独立重做预处理与推理。XGBoost行检查指标和输出有效性，不冒充独立树遍历证明。全部通过。原始结果JSON SHA256：`601e082a6d66275cfc2e5ba02ddea2763ffbd37e9d72acd5a7e65b44b1423164`。

普通测量动态库SHA256：`6ae292799643ca68bb1c203e449e251191c960a067f29c770b85310e2488e18a`。31个运行/SQL/构建/测试/文档文件通过旧、新Git blob及传输SHA256核对后发布。远端PG16/17/18、XGBoost/OpenMP矩阵状态以PR实际检查为准，不由本地测试推断。

## 6. 复现及边界

```sh
# 安装0.5后，在专用可丢弃测试数据库中运行；不能指向生产表。
PG_CONFIG=/path/to/pg_config PGDATABASE=disposable_db \
  python benchmarks/automl/validate.py --inputs /path/to/frozen-inputs \
  --out /absolute/path/evidence --budget 2 --repeats 3
python benchmarks/automl/audit.py /absolute/path/evidence
```

validate.py需要numpy；产品本身不依赖Python。脚本会重建自身的am_bench_input/am_bench_holdout表。完整SQL/API/参数及0.4→0.5升级见../../docs/AUTOML.md。

当前不支持任意关系特征自动生成、CV/时间回测、概率校准、全量refit、跨任务缓存、分布式搜索、只读备机发布或XGBoost外存。GB/TB、并发、峰值内存与数据库干扰尚未形成实测结论。本轮是预算内预测workflow的可运行第一版，不是最终竞争评测。
