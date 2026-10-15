# Source provenance and licenses

## openGauss DB4AI

Original project: https://github.com/opengauss-mirror/openGauss-server
Reference commit: 8f33f38a306dd4b155b6e77d1bc15554b1afcf8f
Areas consulted: src/include/db4ai/, src/gausskernel/dbmind/db4ai/ (excluding snapshots/), and the corresponding training executor code.

Copyright (c) 2020 Huawei Technologies Co.,Ltd.
openGauss is licensed under Mulan PSL v2. The full license is reproduced in LICENSE.
The software is provided on an AS IS basis without warranties, including non-infringement, merchantability or fitness for a particular purpose. See LICENSE for the complete notice.

The new PostgreSQL-only C runtime and numerical adaptation are not represented as an unchanged copy of all original DB4AI implementations. The migration scope is recorded in docs/PORTING.md.

## XGBoost

Project: https://github.com/dmlc/xgboost
Version: 3.4.2
Source release and SHA-256: third_party/xgboost.lock.json
License: Apache License 2.0; see third_party/xgboost/LICENSE and its source notices.

The vendored release also contains required dependency sources, including dmlc-core. Their license/copyright files and upstream notices are retained in their original directories. Vendoring changes distribution and build dependencies; it does not remove upstream attribution or create a new license for those components.
