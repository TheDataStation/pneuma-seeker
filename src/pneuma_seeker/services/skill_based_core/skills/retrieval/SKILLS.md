# Retrieval Skills

## retrieve_tables

Search the data catalog for tables semantically relevant to the user's question.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompts` | list[str] | yes | One or more semantic search queries. Each should combine an entity and a qualifier (e.g. `"high-priority customer orders 2024"`). Use multiple prompts when the question spans different concepts. |

**Returns:** Retrieved table IDs with schemas, sample rows, and detected join paths.

**Notes:**
- Calling twice with rephrased prompts rarely returns new tables. If the second call returns
  the same results, stop retrieving and use `probe_table` or `enumerate_tables` instead.
- Retrieval is not perfect — a table may look irrelevant from column names but actually
  contain relevant data. Always `probe_table` before dismissing a table.

---

## enumerate_tables

Find table IDs matching naming patterns (substring or glob). Use when you suspect multiple
tables cover the same domain (e.g. one table per month, per region, per year).

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patterns` | list[str] | yes | Substring or glob patterns matched against table IDs (e.g. `["sales_2024*", "*_monthly"]`). |

**Returns:** List of matching table IDs.

**Notes:**
- Use after `retrieve_tables` when the user's question spans a time range or multiple
  partitions and the retrieved set appears incomplete.
