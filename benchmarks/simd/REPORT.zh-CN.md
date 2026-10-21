# pg_db4ai 原生 SIMD review 与同机消融

日期：2026-10-19。基于已合并 main `9053edbe6154ed116565d762e41d7abeb9941d9e`。
本轮没有执行外部基线或旧插件。复用已有8组公开数值数据、3个冻结划分，
只运行当前同一个动态库的 off / portable / auto 三种模式。

## Review 结论

DB4AI 主体在已约定范围内基本迁移完成；ABO、snapshots 不属于迁移范围。
裸顶层 CREATE MODEL/PREDICT BY、异步双缓冲、XGBoost 外存训练仍是已有明确边界。
本轮不把 RNG、KMeans/PCA 数值细节当成缺失功能重新改写。

0.3 已有连续数值存储与特征缓存，但未显式加入 SIMD pragma、数值归约约定、
运行时 ISA 分派或向量化诊断。不能称其“完全没有 SIMD”：PGXS 继承的编译
选项可以使部分循环自动向量化。原 OpenMP 开关仅用于 XGBoost 训练线程。

## 补入的实现

私有 C 数值内核覆盖点积、AXPY、L1/L2/L2²/Linf 距离、PCA 居中点积及 Gaussian
映射的矩阵乘法部分，接入 GD/NGD/OVA、随机 PCA、KMeans 和原生预测。
Polynomial 随机映射复用点积内核。通过 omp simd/local reduction 给出编译语义；
-fopenmp-simd 不产生工作线程，也不新增 libgomp/libomp 依赖。

GCC/Clang x86 支持 CPU/OS 检查后的函数级 AVX2；不加全局 -march=native、
-mavx2、AVX512 要求。portable 为编译器基线 ISA；off 保留有序 scalar-source
归约，但不禁用编译器本身的自动向量化。小于16维的点积/AXPY走直接小循环。

Gaussian 权重实际布局为 weights[feature][component]。将跨步读取改为顺序遍历
component，既改善连续访存，又在独立输出之间使用 SIMD；每个输出的 feature
累加顺序不变。点积归约允许局部浮点重排，所以不承诺模型字节/跨编译器逐位一致。

restrict 只用于确实不重叠的写/读缓冲，读只点积允许自别名。不假设 PG 分配或
奇数行宽具备32字节对齐，不使用伪 aligned/assume_aligned。保留安全尾循环。
不启用全局 fast-math/Ofast/finite-math-only；数字内核拒绝这些编译设置，并禁用
FP contraction。NaN/Inf/溢出检查仍在PG层执行，不在SIMD区域调用PG API。

没有修改 vendored XGBoost，也没有声称树遍历/贝叶斯图遍历全部向量化。
cos/exp/log/hypot 仍是 libm 标量调用；稳健归一化和有依赖的状态更新没有强加
ivdep。AVX512/SVE、vector libm、ARM/macOS运行实测和全系统最优性不在本轮结论内。

## 实际测量

Linux x86-64，AMD EPYC 9V74，PostgreSQL18.0，GCC14.2；native-only动态库。
PG postmaster固定到同一个CPU，JIT和查询并行关闭。同一模型/方法每种模式
预热1次，测7次，模式次序在每个重复中固定种子随机化。没有同时运行编译或
其他benchmark。8数据集×3划分，共60个监督学习任务、180个任务/模式组合。

SQL训练包括输入/准备/训练/诊断/持久化；预测包括模型查找/加载、测试表扫描、
预测及SUM，不包括psql进程启动、COPY、预热。预测三种模式使用同一个off训练
的模型。质量按每个划分/模式最后训练的模型计算，不把7次计时当成独立划分。

下面为每划分中位数再跨划分取中位数；加速比先在同机每个重复配对，再汇总，
不一定等于表中两个跨划分中位数直接相除。耗时单位ms。

|任务|off训练|auto训练|配对训练加速|
|---|---:|---:|---:|
|Breast Cancer Logistic GD|10.284|8.619|1.180×|
|Breast Cancer Logistic NGD|10.282|8.713|1.174×|
|Breast Cancer linear SVM|5.689|4.523|1.204×|
|Breast Cancer Gaussian SVM|32.363|17.762|1.836×|
|Banknote Gaussian SVM|57.726|29.374|1.943×|
|Digits OVA Logistic|443.021|238.577|1.842×|
|Digits OVA SVM|277.543|115.690|2.437×|
|California linear GD|80.812|81.445|0.999×|

低维表多数结果在1倍附近，不能宣称全部提升。California只有8个特征，包含bias
仍落在直接小循环，几乎没有模式差异。预测通常亚毫秒，SQL固定开销占比更高；
Digits OVA预测约1.11倍，其他不少配置的微小增减属于需要谨慎解释的噪声。
完整20组配置均保留于summary.csv，不删除不利结果。

补充Digits-42无监督对照（7次，非额外3划分）：KMeans训练23.480→14.838ms，
配对1.564倍；随机PCA训练28.759→17.598ms，配对1.625倍。对应预测/变换约
1.10/1.11倍。KMeans分配完全一致，PCA输出最大差约7.11e-15。

## 数学内核微基准

同一CPU、同一C程序、预热驻留输入，包含函数分派，9次重复，缓冲故意错开
32字节对齐。65维点积44.94→10.86ns（4.14倍），L2²距离47.79→11.59ns
（4.12倍），居中点积43.96→15.03ns（2.92倍），AXPY13.00→9.33ns（1.39倍）。

Gaussian-linear(32输入、129输出)1733.89→386.32ns（4.49倍）；1024输出时约
19.30倍，包含缓存/访问顺序改善，且不包括cos，绝不能写成整个训练提升19倍。
微基准展示的是具体内核，不是SQL、I/O、树模型或并发服务吞吐。

## 一致性与验证

60组监督任务，auto与off训练后的分类标签/回归预测值均相同；这不表示权重
字节或不同模型的原始分类margin逐位相同。同一off模型在模式间切换时，原始
预测最大差1.07e-14。PCA变换最大差7.11e-15，KMeans分配无变化。

GCC14和Clang17分别通过39519项独立数值检查，含0/奇数/向量边界维数、非32B
对齐、只读输入自别名、guard-page精确尾部、NaN/Inf/溢出。GCC ASan+UBSan
组合也通过这套检查；没有据此宣称整个数据库通过ASan。

PG18 native完整集成套件在auto、运行时off、编译WITH_SIMD=0三种配置通过，
覆盖数值oracle、部分batch、缓存、shuffle/溢写、DDL/RLS、升级/备份恢复。
GCC与Clang向量化报告、GCC反汇编确认有真实向量循环和AVX2 vaddpd/vmulpd，
不是只有pragma。WITH_SIMD开关无需clean就能触发重编译；还原后的动态库SHA256
与测量库相同：9aa03c834c0f333a62b8e067529d73672bf04b6a77fcf3bd3acc058df6a90c17。

原始日志、SQL、预测CSV、编译器报告/反汇编、逐次耗时和源文件指纹在证据包中。
代码没有新增PG模型格式，因此原0.3安装替换动态库后需新建连接，无需伪造SQL
升级。复现方式见README.md，数值与部署约束见../../docs/SIMD.md。
