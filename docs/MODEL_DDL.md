# Model DDL compatibility entrypoint (0.3)

**Stock PostgreSQL 16–18 does not accept bare `CREATE MODEL` from this extension.**
The raw parser calls its built-in grammar before ProcessUtility hooks. Registering
a utility hook or CustomScan cannot make an unknown top-level statement parse.
See the upstream [parser](https://github.com/postgres/postgres/blob/REL_18_STABLE/src/backend/parser/parser.c).
No PostgreSQL grammar patch, wire proxy or hidden parser replacement is included.

Instead, the extension implements the model language at an explicit SQL entrypoint:

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

`db4ai.parse_ddl(text)` is a dry parser returning a structured request and generated
training SELECT without querying data or training. `db4ai.build_info()` explicitly
reports `native_create_model_grammar:false`. Bare SQL is rejected by stock PG.
A client adapter could wrap model DDL in execute_ddl, but that is client-side
compatibility, not a changed PostgreSQL wire grammar.

The interpreter uses PG's lexical scanner, not regex. Quoted UTF-8 identifiers,
comments, strings, dollar quoting, nested subqueries and expression features are
handled with token/depth boundaries. PG parses the reconstructed SELECT and
performs normal access checks through the existing training API.

## Accepted shape

```
CREATE MODEL name USING algorithm
 FEATURES expression [,expression...]
 [TARGET expression]
 FROM relation_or_parenthesized_query [WHERE ...] [ORDER BY ...]
 [WITH key=literal_or_DEFAULT [,key=literal_or_DEFAULT...]]
DROP MODEL [IF EXISTS] name
EXPLAIN MODEL name
```

Supervised training requires one TARGET expression and explicit FEATURES to avoid
accidental target leakage. Unsupervised algorithms reject TARGET and can omit
FEATURES to select all source fields. Parameters accept numerical/string/boolean
literals and DEFAULT; unknown and inapplicable keys are rejected, including with
DEFAULT. Default profile is `opengauss`; explicit profile can override it.
`model_format='json'/'ubj'` selects durable model IO; UBJSON is XGBoost-only.

Only one model command per call is allowed. Model names are not PG schema objects;
this API takes one identifier, not an arbitrary schema-qualified object name.
Unicode U&-escaped model identifiers and full openGauss EXPLAIN option/format syntax
are not supported; use ordinary UTF-8 quoting and the returned JSON diagnostics.
`PREDICT BY` expression grammar is not intercepted; typed db4ai.predict_* functions
provide prediction. Use ordinary PG EXPLAIN for prediction SELECTs.

## Security and transactions

The API is SECURITY INVOKER. It passes parameters to fixed schema-qualified SPI
calls to db4ai.train/drop_model/explain_model. It does not elevate training SQL.
Source permissions/RLS and underlying function EXECUTE grants apply. Failed
commands and enclosing transaction rollback do not publish partial models. The
scanner is bounded to 16 MiB / 262144 tokens / 256 nested delimiter levels.

Tests include semicolon injection, comments/dollar strings, duplicate options,
invalid targets, malformed signs, dry-run nonexecution, permission revocation,
source RLS, missing models, IF EXISTS and rollback.
