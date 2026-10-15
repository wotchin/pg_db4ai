# pg_db4ai：PostgreSQL 内置机器学习扩展

这是一个面向 PostgreSQL 的 0.1 版实现。自己的代码为 C11，直接使用 PG 的 SPI、MemoryContext、TupleTableSlot、tuplestore 和错误处理机制，不设计数据库无关层，不修改 PG 内核。

**当前版本已打通训练、模型保存、跨连接加载、预测、权限隔离和备份恢复，但不是 openGauss DB4AI 全部功能的等价移植。** 原版优化器、初始化方式、类别类型等差异见 [迁移边界](docs/PORTING.md)。

## 已实现

| 部分 | 实现 |
|---|---|
| 模型表 | `db4ai.pg_db4ai_models`，普通 WAL 持久化扩展表 |
| 原生算法 | 线性回归、二分类逻辑回归、线性 SVM、KMeans、PCA |
| XGBoost | 平方误差回归、二分类 logistic、logistic 回归、gamma 回归 |
| 数据输入 | SPI 分批取数，转换后写入可溢写 tuplestore，多轮训练不重跑源 SQL |
| 持久化 | 原生模型为显式版本化 JSON；XGBoost 默认 UBJSON，也支持 JSON |
| 权限 | 源查询以调用者身份执行；模型按当前角色隔离，普通用户不能直接改模型二进制 |
| 事务 | 训练成功后才保存模型；支持回滚及事务内删除后重建 |

XGBoost 3.4.2 及其随附依赖源码已内置于 `third_party/xgboost`，有发布包 SHA-256 和逐文件清单。正常编译不联网拉源码，不需要 submodule、pip 或外部 `libxgboost.so`。适配器为纯 C；XGBoost 上游内部仍是 C++，通过 C API 调用。其静态库链接进扩展，但启用它时仍需 C++ 编译器及系统 C++ 运行库。

## 编译安装

目前验证平台为 Linux x86-64；不同 PostgreSQL 主版本需要分别编译。

```sh
git clone https://github.com/wotchin/pg_db4ai.git
cd pg_db4ai
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 XGB_JOBS=2
sudo make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 install
# 以普通操作系统用户运行，脚本会建立临时测试数据库实例。
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 test
```

关闭 XGBoost 可使用 `WITH_XGBOOST=0`，此时扩展不依赖 C++ 运行库。切换该选项或 PG 版本前先执行 `make clean`。

## 可执行示例

```sql
CREATE EXTENSION pg_db4ai;
CREATE TABLE public.db4ai_demo AS
SELECT 1 + 2*x - 3*x*x AS y, x, x*x AS x2
FROM (SELECT i::float8/50 AS x FROM generate_series(-50,50) AS i) s;

SELECT db4ai.train('demo', 'linear_regression',
    'SELECT y,x,x2 FROM public.db4ai_demo ORDER BY x',
    '{"max_iterations":1500,"learning_rate":0.15,"tolerance":1e-12}');
SELECT db4ai.predict_float8('demo', ARRAY[0.25,0.0625]);
-- 结果应接近 1.3125。
SELECT db4ai.explain_model('demo');

SELECT db4ai.train('demo_xgb', 'xgboost_regression_squarederror',
    'SELECT y,x,x2 FROM public.db4ai_demo',
    '{"max_iterations":80,"learning_rate":0.15,"max_depth":3}', 'json');
SELECT db4ai.predict_float8('demo_xgb', ARRAY[0.25,0.0625]);
SELECT convert_from(db4ai.export_model('demo_xgb'), 'UTF8')::jsonb;
```

监督学习输入的第一列是标签，后续为按顺序排列的标量特征；KMeans/PCA 输入只有特征。当前支持数值及布尔列，二分类标签为 0/1。NULL、NaN 和无穷值被明确拒绝，预处理由 SQL 显式完成。

## 测试与边界

本地 PostgreSQL 18.0 上，开启和关闭 XGBoost 的两套完整集成测试均通过。覆盖数值结果、JSON/UBJSON 一致性、跨连接加载、角色隔离、回滚、模型切换缓存、低内存溢写、源 SQL 只执行一次、超时取消及 pg_dump/pg_restore。源码已经按 Git blob SHA 与本地测试版本逐个核对。

CI 配置了 PG 16、17、18 以及 PG 18 纯 C 构建；以具体提交的 Actions 结果为准，不把“配置了测试矩阵”等同于“所有版本已通过”。

尚未提供原版所有求解器、任意类型分类标签、多分类/Bayes、旧 openGauss 模型导入、并行/分布式训练、XGBoost 外存训练。KMeans 初始化及 PCA 求解器与原版不同，不能视为完全相同的算法实现。

`db4ai.xgboost_max_input_mb` 只限制输入复制，不是 XGBoost 进程总内存上限。取消在上游调用之间检查，不能保证正在执行的单次 XGBoost 调用立即中止。尚未做生产负载性能基准。

详细说明：[英文 README](README.md)、[架构](docs/ARCHITECTURE.md)、[迁移差异](docs/PORTING.md)、[验证记录](docs/VALIDATION.md)、[许可证与来源](THIRD_PARTY_NOTICES.md)。
