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
        semantic_ops: list[str] = []
        if self.config.ENABLE_SEMANTIC_JOIN:
            semantic_ops.append(
                f"- **{ActionNames.SEMANTIC_JOIN.value}**: Joins two tables by computing semantic similarity between specified columns."
            )
        if self.config.ENABLE_SEMANTIC_COL_GEN:
            semantic_ops.append(
                f"- **{ActionNames.SEMANTIC_COLUMN_GENERATION.value}**: Adds a new column to a table using an LLM."
            )

        semantic_section = ""
        if semantic_ops:
            semantic_section = (
                f"\n  Aside from relational operations, {ActionNames.MATERIALIZER.value} "
                "also supports the following semantic operations:\n"
                + "\n".join(f"  {op}" for op in semantic_ops)
            )

        return (
            f"**{ActionNames.MATERIALIZER.value}**:\n"
            f"    Populate tables in T with rows derived from data integration and processing.{semantic_section}\n"
            '    - **Args**: {"note": "<additional note or empty string>"}'
        )
