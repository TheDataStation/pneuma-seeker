from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class MaterializerInvocation(Action):
    """Pseudo-action that instructs Conductor to invoke the Materializer sub-agent."""

    action_name = ActionNames.MATERIALIZER
    agents = frozenset({AgentType.CONDUCTOR})
    flag = None
    order = 4
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        extra_ops: list[str] = []
        if self.config.ENABLE_SEMANTIC_JOIN:
            extra_ops.append(
                f"- **{ActionNames.SEMANTIC_JOIN.value}**: Joins two tables by computing semantic similarity between specified columns."
            )
        if self.config.ENABLE_SEMANTIC_COL_GEN:
            extra_ops.append(
                f"- **{ActionNames.SEMANTIC_COLUMN_GENERATION.value}**: Adds a new column to a table using an LLM."
            )
        extra_ops.append(
            f"- **{ActionNames.ENTITY_RESOLUTION.value}**: Harmonizes noisy text values in a column by producing an `original_value → canonical_value` mapping table (supports both unsupervised clustering and supervised matching against a seed list)."
        )

        extra_section = (
            f"\n  Aside from relational operations, {ActionNames.MATERIALIZER.value} "
            "also supports the following operations:\n"
            + "\n".join(f"  {op}" for op in extra_ops)
        )

        return (
            f"**{ActionNames.MATERIALIZER.value}**:\n"
            f"    Populate tables in T with rows derived from data integration and processing.{extra_section}\n"
            '    - **Args**: {"note": "<integration hints or empty string>", "mode": "<optional: \'fresh\' (default) | \'update\' | \'reset\'>"}\n'
            "    - **mode values**:\n"
            "      - `fresh` (default, omit to use): first-time materialization; Materializer starts from a clean slate.\n"
            "      - `update`: T schema changed slightly (e.g., added a column, remapped values); Materializer reuses intermediate tables from the prior run and applies only the delta. Use the `note` arg to describe exactly what changed.\n"
            "      - `reset`: major T redesign; Materializer discards all prior intermediate tables and starts from scratch even if a prior materialization exists."
        )
