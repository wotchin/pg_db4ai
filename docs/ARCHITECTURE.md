# PostgreSQL-only architecture

```
SQL functions (SECURITY INVOKER)
  |-- train
  |    |-- parse/check one SELECT
  |    |-- SPI cursor under caller identity and RLS
  |    |-- numeric tuple conversion -> tuplestore (memory + spill)
  |    |-- repeated passes -> native C algorithms / XGBoost C API
  |    `-- durable artifact -> parameterized protected registry INSERT
  |-- predict_float8 / predict_bool / predict_cluster / transform
  |    `-- owner-visible model -> statement/command/role/name cache -> predictor
  `-- explain_model / export_model / drop_model

db4ai.pg_db4ai_models
  identity + regrole owner + metadata + effective options + metrics + bytea artifact
```

## Ownership and transactions

The model table is owned by the extension installer. Regular users have SELECT through an owner-filtering RLS policy, not arbitrary artifact-write privileges. Training remains invoker-security; only fixed registry INSERT/DELETE statements run with the extension owner's identity. Every such operation saves and restores identity in both normal and PG_CATCH paths and always filters/specifies the original caller's role. No input training SQL is passed to these privileged blocks. The fixed extension schema is not writable by PUBLIC.

Training publishes only after a complete model has been produced. Publication uses ordinary transactional table writes; there are no sidecar model files or custom WAL. Names are unique per role. Replacement is an explicit transaction containing drop plus train. Cross-role sharing and model ACL grants are not currently provided.

The RLS policy uses current_user, not session_user. SECURITY DEFINER callers intentionally inherit their definer's data/model identity. Administrators must handle dropped-role orphan models; regrole values alone do not implement a new native PostgreSQL object-dependency class.

## Memory and errors

Training gets a child memory context. Normalized input uses tuplestore and PG-managed temporary files. Tuples are copied/converted before each SPI batch is released. Predictor contexts hold prepared models and have backend-local lifetimes. XGBoost Booster/DMatrix resources register MemoryContext callbacks, plus explicit cleanup on success and failed prediction paths. Cleanup callbacks do not raise new PostgreSQL errors.

PostgreSQL API functions are never called through callbacks executed by XGBoost's C++ stack. Dense input is prepared before entering the C API. Once a C API function returns, failures are translated into ereport(ERROR). This avoids PostgreSQL longjmp across upstream C++ RAII frames. CPU cancellation is cooperative between upstream calls, not asynchronous unwinding of C++.

The input-memory cap bounds our dense copy only. The native extension is not an OS-process sandbox. No public arbitrary XGBoost model import endpoint is exposed, because malformed third-party model inputs are not a safe database boundary.

## Durable model format

The table's format_version describes the pg_db4ai artifact envelope contract. Native JSON independently includes version, algorithm, dimensions, weights and optional mean. No raw struct layout, pointer, Datum, OID or byte-order-dependent float memory image is persisted in the native model payload.

XGBoost bytea contains exactly the result of XGBoosterSaveModelToBuffer with json or ubj format. Metadata separately captures effective training settings, feature order and the library version. Reload uses XGBoosterLoadModelFromBuffer. Runtime snapshots and human-readable dump APIs are not model persistence mechanisms.

pg_extension_config_dump registers both the model relation and identity sequence. Owner regrole values are represented by role names in logical dump data. Required roles must exist at restore time.
