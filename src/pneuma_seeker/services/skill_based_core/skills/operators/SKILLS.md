# Operator Skills

Operators transform workspace tables without requiring Python or SQL. Prefer operators over
`run_sql` when they express the transformation cleanly — they are simpler and leave a clearer
audit trail. Results are persisted immediately and reusable by subsequent skills.

## project_table

Select and/or rename columns from a source table. Unselected columns are dropped.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `src_table_id` | str | yes | Source table ID (retrieved or workspace). |
| `target_table_id` | str | yes | Output table ID saved to workspace. |
| `column_mapping` | dict[str,str] | yes | `{"source_col": "target_col"}`. Columns not listed are dropped. |

**Returns:** Confirmation with output column list and row count.

---

## join_tables

Inner-join two tables on one or more equal key columns.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `left_table_id` | str | yes | Left table ID. |
| `right_table_id` | str | yes | Right table ID. |
| `left_keys` | list[str] | yes | Key column(s) in the left table. |
| `right_keys` | list[str] | yes | Matching key column(s) in the right table (same length). |
| `output_table_id` | str | yes | Output table ID saved to workspace. |

**Returns:** Confirmation with row count and column list.

---

## union_tables

Stack multiple tables vertically (UNION ALL). All tables must have compatible schemas.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `table_ids` | list[str] | yes | Tables to stack. |
| `output_table_id` | str | yes | Output table ID saved to workspace. |
| `provenance_column` | str | no | Column recording which source table each row came from. Default: `_source`. |
| `provenance_regex` | str | no | Regex applied to source table ID to form the provenance label. Default: `(.*)`. |

**Returns:** Confirmation with row count and column list.
