#!/usr/bin/env bash
# Print rust/crates/hv-core/src/columns.rs from export_data.py's COLUMNS (run from the repo root).
set -euo pipefail
python - <<'PY'
import json, export_data as e
def lit(s): return json.dumps(s, ensure_ascii=False)
print("//! export_data.py's COLUMNS: every table's columns and their descriptions (metadata.json).")
print("//! Generated from export_data.py; regenerate when that changes:")
print("//!     rust/bench/gen_columns.sh > rust/crates/hv-core/src/columns.rs")
print()
print("pub const COLUMNS: &[(&str, &[(&str, &str)])] = &[")
for t, cols in e.COLUMNS.items():
    print(f"    ({lit(t)}, &[")
    for c, d in cols.items():
        print(f"        ({lit(c)}, {lit(d)}),")
    print("    ]),")
print("];")
print()
print("pub fn columns(table: &str) -> Vec<&'static str> {")
print("    COLUMNS.iter().find(|(t, _)| *t == table).map(|(_, c)| c.iter().map(|(n, _)| *n).collect()).unwrap_or_default()")
print("}")
print()
print("pub const EVENT_FIELDS: &[&str] = &[" + ", ".join(lit(f) for f in e.EVENT_FIELDS) + "];")
PY
