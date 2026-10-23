# pg_db4ai：PostgreSQL 原生机器学习扩展

## 0.4：把内部 Bayes 替换为用户级分类器

新增 **`gaussian_naive_bayes`**，支持二分类/多分类、原始类型标签、
`predict_proba` 和 `predict_log_proba`。输入为第一列标签、后续数值特征。
这是真正的高斯朴素贝叶斯分类器，不是给原选择率估计器换名字。

移除 `bayes_net_internal` 训练/推理和新安装的 `predict_joint`；旧模型保留导出/删除，
不会自动变成分类模型，也不会在升级时删除。安装新库并重新连接后升级到 **0.4.0**。
概率是模型后验，不是已经校准的置信度；文本类别特征和通用贝叶斯网络不在支持范围。
[使用、统计假设和升级说明](docs/NAIVE_BAYES.md)

## 历史 0.3 更新

新增 CorgiPile 两级训练混排、连续数值重扫与可溢写的特征预计算缓存；补齐模型 DDL 显式解析入口、KMeans/PCA 诊断、小数 max_seconds，以及可选 `WITH_OPENMP=1` 的 XGBoost 多线程训练。

**ABO 和 snapshots 不在迁移范围。** RNG、KMeans 采样、随机 PCA 和 OVA 状态的差异是允许的实现选择，不再作为功能缺失。

`SELECT db4ai.execute_ddl($ddl$CREATE MODEL ...$ddl$)` 在插件内解析执行；**原生 PG 顶层裸 CREATE MODEL 仍不支持**。`SELECT db4ai.build_info()` 可查看编译能力。

升级：安装新库、重新连接，然后 `ALTER EXTENSION pg_db4ai UPDATE TO '0.3.0';`。

[功能核查](docs/FUNCTIONAL_COVERAGE.md) · [DDL 边界](docs/MODEL_DDL.md) · [论文与 CorgiPile 实现](docs/CORGIPILE.md) · [仅测本插件的实验](benchmarks/native/README.md)


历史 0.2 曾覆盖固定版本 openGauss DB4AI 注册表中的 **11 个算法入口**（0.4 已将内部 Bayes 替换），补入
NGD、One-vs-All 多分类、Gaussian/Polynomial 核特征映射、随机 PCA、KMeans
两类初始化与四种距离、类型化标签、Bayes network 和训练 shuffle cache；
另提供真正的 **PG CustomScan `ShuffleScan`**。

自己的代码全部为 **C11**，直接使用 PG 的 SPI、MemoryContext、Tuple、tuplestore
和错误处理。模型表是 **`db4ai.pg_db4ai_models`**，不改 PG 内核、不设计数据库无关层。
XGBoost 3.4.2 及依赖源码内置，静态链接，通过纯 C API 调用；上游内部仍是 C++。

这里的“11 个算法入口齐全”不表示已经与原版运行中的 openGauss 做完全差分验收，
也不保证相同种子会逐位产生相同结果。[迁移边界](docs/PORTING.md)明确列出差异。

## 编译与升级

```sh
git clone https://github.com/wotchin/pg_db4ai.git
cd pg_db4ai
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 XGB_JOBS=2
sudo make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 install
# 非 root 用户运行，建立临时测试实例：
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 test
```

安装目标 PG 开发头文件、C 编译器、GNU Make；启用 XGBoost 还需要 CMake/C++17。
正常构建不联网下载源码、不使用 submodule/pip，不需外部 `libxgboost.so`。
`WITH_XGBOOST=0` 包含全部原生算法，且不依赖 C++ 运行库。
切换 PG 版本或此选项前执行 `make clean`。

```sql
CREATE EXTENSION pg_db4ai; -- 新安装
-- 已安装 0.1，安装新库和 SQL 脚本后：
ALTER EXTENSION pg_db4ai UPDATE TO '0.2.0';
```

替换动态库后请重新连接，已加载的 C 代码不会被 ALTER EXTENSION 热替换。
升级不改写已有模型；0.1 原生 JSON 仍可预测，但其旧分类模型没有标签字典，
新增类型化标签接口需要重训。这里不支持直接导入旧 openGauss 二进制模型。

## 1. GD / NGD 与训练 ShuffleScan

NGD 是原版的 **Normalized Gradient Descent（归一化梯度下降）**，不是自然梯度。
监督训练第一列为标签，后续为特征。特征可以是数值/布尔标量，也支持定长
`float4[]/float8[]`，按列展开。预测特征顺序必须一致。

```sql
CREATE TABLE public.regression_demo AS
SELECT 1+2*x-3*x*x AS y, x, x*x AS x2
FROM (SELECT i::float8/50 AS x FROM generate_series(-50,50)i)s;
SELECT db4ai.train('reg', 'linear_regression',
    'SELECT y,x,x2 FROM public.regression_demo ORDER BY x',
    '{"optimizer":"ngd","max_iterations":100,"learning_rate":0.1,
      "batch_size":16,"shuffle":"batch","shuffle_buffer_size":8,"seed":42}');
SELECT db4ai.predict_float8('reg', ARRAY[0.25,0.0625]);
SELECT db4ai.explain_model('reg');
```

默认 `profile:"stable"` 保留保守学习率、严格 NULL 检查及 Jacobi PCA。
`profile:"opengauss"` 选择原版风格的 GD 默认参数、legacy NULL 策略及不居中的随机 PCA；
它不是“结果逐位一致”开关。明确传入的参数覆盖 profile。

训练源 SQL **只执行一次**，之后重扫可溢写 tuplestore。第 0 轮默认顺序读取；
后续轮次整批缓存洗牌，内存不足时有界随机替换并随机排空，保证每轮不漏不重。
整份数据可缓存时批次排列均匀；有界模式不是全局均匀洗牌。
`shuffle:"row"` 先按行洗牌，再重新组装指定大小的训练批次；`shuffle:"none"` 关闭乱序。
固定 seed 还不够，复现实验也应固定源 ORDER BY、batch/cache 大小和软件版本。

## 2. 类型化标签与多分类

```sql
CREATE TABLE public.classes_demo(label text, x0 float8, x1 float8, x2 float8);
INSERT INTO public.classes_demo VALUES
  ('apple',1,0,0),('pear',0,1,0),('peach',0,0,1);
SELECT db4ai.train('fruit', 'multiclass',
  'SELECT * FROM public.classes_demo ORDER BY label',
  '{"classifier":"logistic_regression","optimizer":"gd","max_iterations":200,
    "learning_rate":0.3,"batch_size":3,"tolerance":0}');
SELECT db4ai.predict_text('fruit', ARRAY[0.0,1.0,0.0]);
SELECT db4ai.predict_label('fruit', ARRAY[0.0,1.0,0.0], NULL::text);
SELECT db4ai.class_labels('fruit'), db4ai.predict_scores('fruit', ARRAY[0.0,1.0,0.0]);
```

多分类采用 One-vs-All，基分类器可选 logistic 或 SVM，并支持 NGD。
标签通过 PG 类型的 hash/equality 保存，不把 bigint/numeric 标签先转换成浮点数。
`predict_label` 的第三参数提供返回类型；枚举、text、numeric 等使用对应的 `NULL::类型`。
域类型展开为底层类型；不支持 pseudo-type/匿名 record。
OVA logistic 分数不是 softmax，SVM 分数是 margin。二分类 0/1 和 bool 标签规范为 0,1 顺序；
其他类型按首次出现顺序映射，`class_labels` 可检查。

## 3. 非线性 SVM 与随机 PCA

```sql
SELECT db4ai.train('nonlinear', 'svm_classification',
  'SELECT (x*y<0)::int, x::float8, y::float8
   FROM (VALUES(-1),(1))a(x) CROSS JOIN (VALUES(-1),(1))b(y)',
  '{"kernel":"gaussian","components":128,"gamma":0.5,"optimizer":"ngd",
    "max_iterations":200,"learning_rate":0.3,"tolerance":0}');
-- polynomial 使用 degree/coef0/components；真实映射随模型持久化。
SELECT db4ai.train('pca_stochastic', 'pca',
  'SELECT i::float8, 2*i::float8 FROM generate_series(-30,30)i',
  '{"profile":"opengauss","number_components":1,"max_iterations":20,
    "batch_size":8,"shuffle":"row","tolerance":0}');
SELECT db4ai.transform('pca_stochastic', ARRAY[1.0,2.0]);
```

PCA 可以显式 `pca_solver:"stochastic"` / `"jacobi"` 和 `center:true/false`。
Jacobi 不执行训练乱序，显式提供 shuffle 参数会报错，避免参数被悄悄忽略。
KMeans 支持 `seeding_function:"Random++"/"KMeans||"/"reservoir"`、
`distance_function:"L1"/"L2"/"L2_Squared"/"Linf"`，以及 `num_centroids`。

## 4. Gaussian Naive Bayes 分类

```sql
SELECT db4ai.train('classifier', 'gaussian_naive_bayes',
  'SELECT label,x,z FROM public.training_data');
SELECT db4ai.predict_label('classifier', ARRAY[1.0,2.0], NULL::text);
SELECT db4ai.class_labels('classifier'),
       db4ai.predict_proba('classifier', ARRAY[1.0,2.0]);
```

类别顺序用 `class_labels` 查询；后验概率之和为1，但不能当作已校准置信度。
原生算法代码仍是 C11，使用现有 PG 权限、事务、类型系统和 SIMD 内核。

## 5. 可以被 EXPLAIN 看到的 ShuffleScan

```sql
LOAD 'pg_db4ai'; -- 在该连接规划查询之前加载 hook
EXPLAIN (ANALYZE, COSTS OFF)
SELECT x*2 FROM db4ai.shuffle_scan(
  'SELECT i FROM generate_series(1,100)i', 42, 16, 1
) AS s(x integer) WHERE x>10;
```

适用的常量参数调用显示 **`Custom Scan (ShuffleScan)`**，支持过滤、投影、重扫及
执行统计。`epoch=0` 顺序扫描，`epoch>=1` 乱序；重扫复用同一份数据，同一 seed/epoch
保持相同排列。普通 EXPLAIN 不执行源 SQL。
动态参数、LATERAL、whole-row/ordinality 等情况可能走普通 FunctionScan 回退，语义不变。
函数 EXECUTE 权限和源表 RLS 都被检查；不用修改 PG grammar，也不要求 shared_preload。
这与训练内部的批量 shuffle 是两个接入面，训练不必逐行调用 SQL 函数。

## 持久化、测试与范围

模型表仍为 `db4ai.pg_db4ai_models`。原生模型使用版本化 JSON，包含标签类型/值、
权重、实际随机核映射、PCA 均值/基向量/特征值估计；不保存内存地址或裸 Datum。
XGBoost 四种原有 objective 使用 JSON/UBJSON model IO，默认 UBJSON。
普通用户不能直接修改模型二进制，源 SQL 以调用者身份执行，模型保存支持事务回滚。

本地 Linux x86-64 / PG18.0 的 WITH_XGBOOST=1 和 =0 完整套件均通过，覆盖原有回归、
新增算法、手算/独立数学 oracle、ShuffleScan 实际计划、权限/RLS、重扫/溢写、
0.1→0.2 升级、跨连接加载及 pg_dump/pg_restore。PG16/17/18 远端结果应看对应提交的 CI，
不能把“配置测试矩阵”当成“全部测试已通过”。

仍未验证运行中 openGauss 的全功能逐位等价、相同随机种子的逐位一致性、生产性能、
Windows/macOS、并行/分布式训练、XGBoost 外存训练和旧 openGauss 二进制模型导入。
资源预算是每个 tuplestore/cache/模型的预算，不是整个 backend 的总内存硬限制。
XGBoost 取消在上游调用之间检查。原始 snapshots 数据版本功能仍不在范围内。

[英文 API 与完整约束](README.md) · [架构](docs/ARCHITECTURE.md) ·
[原版映射与差异](docs/PORTING.md) · [验证记录](docs/VALIDATION.md) · [更新记录](CHANGELOG.md)
