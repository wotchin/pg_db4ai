# pg_db4ai

PostgreSQL-native, in-process machine learning extension derived from the DB4AI design in openGauss.

## Project boundaries

- Target PostgreSQL extensions, not a database-independent runtime.
- First-party implementation is C and uses PostgreSQL memory contexts, SPI, tuples and error handling.
- Model relation: `db4ai.pg_db4ai_models` (an ordinary WAL-logged extension table, not a patched system catalog).
- No changes to the PostgreSQL parser or executor are required.
- Vendor the official XGBoost source distribution, including its required source dependencies, with a pinned SHA-256. Normal builds must not fetch repositories.
- Keep upstream XGBoost as C++; call only its C API from this extension. Do not rewrite upstream STL internals.
- Persist XGBoost models with model IO (`XGBoosterSaveModelToBuffer` / `XGBoosterLoadModelFromBuffer`), not memory snapshots or display-only dumps.
- Exclude the openGauss DB4AI `snapshots/` feature.

Implementation, build instructions and tested support boundaries are added with the extension sources.
