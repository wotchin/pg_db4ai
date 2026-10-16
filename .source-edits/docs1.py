FILES['docs/PORTING.md'] = {'old': '80a905e5d48a70898341a9f82074edb26ebb2111', 'new': 'edf84ba5524fde0ead581cded9d380c6f9974c02', 'base': 'docs/PORTING.md', 'mode': '100644','edits':[
(0,0,r'''# Scope update for 0.3

The detailed numerical differences below are accepted implementation choices,
not unresolved functional migration gaps. ABO optimizer integration is excluded
by project scope, as is snapshots. See [FUNCTIONAL_COVERAGE.md](FUNCTIONAL_COVERAGE.md)
for the current capability audit. Model diagnostics, fractional time limits,
optional XGBoost threads and explicit model-DDL interpretation are now included.
The new CorgiPile training iterator and dense/prepared replay are described in
[CORGIPILE.md](CORGIPILE.md); native top-level grammar remains outside ordinary
PG extension hooks, documented in [MODEL_DDL.md](MODEL_DDL.md).

## Historical numerical reference (0.2)

'''),
(85,87,r'''   C++/standard runtime remains; GPU/distributed/external-memory training is
   not enabled. OpenMP is now optional in 0.3; default nthread is 1; training metrics are not held-out validation.
''')
]}
