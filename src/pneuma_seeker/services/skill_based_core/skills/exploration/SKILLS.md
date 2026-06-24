# Exploration Skills

## situational_analysis

Think step-by-step about the current situation before acting. Use this at the start of each
turn and whenever you need to reason through a decision.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | str | yes | Your analysis: what you know, what is still missing, and what the next skill should be. |

**Returns:** Acknowledgement (no side effects).

---

## probe_table

Ask factual questions about the actual contents of retrieved tables (context extraction).
Always probe before concluding a table is irrelevant based on column names alone.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uncertainties` | list[dict] | yes | Each item: `{"table_ids": ["<id>", ...], "question": "<question>"}`. Questions should be specific (e.g., "What distinct values are in column X?", "What is the row count?"). |

**Returns:** A structured summary of findings.

**Notes:**
- Use to validate value encodings before using a column as a filter or join key.
- Use to confirm a table contains the time range or domain the user asked about.
- Multiple questions can be batched in a single call.
