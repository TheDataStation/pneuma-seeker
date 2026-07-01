# Definition Skills

## define_target

Declare the target tables (T) and analysis script (S) that represent the user's information
need. This is the **relational reification** step — it makes the agent's intent explicit and
visible to the user and persists it across turns.

**Call this before using any operator or execution skill.** It is the SkillsAgent equivalent
of the Conductor's `state_manipulation` action.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `T` | dict[str, list[str]] | no | Target table schemas: `{"table_id": ["col1", "col2", ...]}`. Use descriptive, semantically clear table and column names. Prefer a single unified table unless the analysis genuinely requires separate views. |
| `column_descriptions` | dict[str, dict[str, str]] | required if T given | Per-column natural-language descriptions: `{"table_id": {"col": "what it means"}}`. Required when T is provided. |
| `S` | str | no | Python analysis script to run over materialized T. Should only reference tables in T. Write it after the table schemas are clear. |

**Returns:** Confirmation of what was set.

**Notes:**
- T defines *what schema* to produce. Operator skills (`join_tables`, `project_table`, etc.)
  and `run_sql`/`run_python` populate the actual rows.
- S is executed *after* T is populated via `run_python` (pass the S code as the `code` arg).
- Prefer a **single unified table** in T. Multiple tables are only warranted when the analysis
  genuinely requires separate independent views.
- Include in T only columns that S directly uses. Do not add speculative "context" columns.
- You may call `define_target` again to refine T or S as you learn more from probing.
