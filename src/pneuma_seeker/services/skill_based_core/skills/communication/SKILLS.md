# Communication Skills

## respond

Send the final answer to the user. **This ends the current turn.**
Call this when you have gathered enough information to answer the user's question.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | str | yes | The response to show the user. Be concise. Disclose any proxies or data limitations. |

**Returns:** Confirmation (the loop terminates immediately after this skill is called).
