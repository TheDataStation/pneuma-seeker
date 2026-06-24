from pandas import DataFrame

from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Table


class DefineTargetSkill(SkillBase):
    """Relational reification — declares (T, S) on the agent state.

    This mirrors Conductor's state_manipulation action. It is a pure in-memory
    schema operation: no DB tables are created or dropped.
    """

    name = "define_target"

    def execute(self, args: dict, agent) -> SkillResult:
        T_schema: dict | None = args.get("T")
        column_descriptions: dict | None = args.get("column_descriptions")
        S: str | None = args.get("S")

        if T_schema is not None and column_descriptions is None:
            return SkillResult(
                "Error: 'column_descriptions' is required when 'T' is provided."
            )

        modified: list[str] = []

        if T_schema is not None:
            T_docs: dict = {}
            for table_id, columns in T_schema.items():
                T_docs[table_id] = Table(
                    doc_id=table_id,
                    retriever_type=RetrieverType.CONDUCTOR,
                    content=DataFrame(columns=columns),
                    metadata={},
                    path=table_id,
                )
            agent.state.T = T_docs
            agent.state.column_descriptions = column_descriptions  # type: ignore[assignment]
            agent.state.is_T_materialized = False
            modified.append("T")

        if S is not None:
            agent.state.S = S
            agent.state.is_S_executed = False
            modified.append("S")

        if not modified:
            return SkillResult("Error: at least one of 'T' or 'S' must be provided.")

        return SkillResult(f"Target defined: {', '.join(modified)} set.")
