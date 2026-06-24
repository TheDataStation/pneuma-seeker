from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult


class SituationalAnalysisSkill(SkillBase):
    name = "situational_analysis"

    def execute(self, args: dict, agent) -> SkillResult:
        message = args.get("message", "")
        if not message:
            return SkillResult("Error: 'message' is required.")
        return SkillResult(f"Acknowledged: {message}")


class ProbeTableSkill(SkillBase):
    name = "probe_table"

    def execute(self, args: dict, agent) -> SkillResult:
        uncertainties = args.get("uncertainties")
        if not isinstance(uncertainties, list) or not uncertainties:
            return SkillResult(
                "Error: 'uncertainties' must be a non-empty list of "
                "{table_ids, question} objects."
            )
        available = agent.retrieved_tables + agent.external_tables
        result_table = "skills_probe"
        try:
            summary, log_msgs = agent.action_set.run_context_extraction(
                uncertainties, available, result_table
            )
            for m in log_msgs:
                agent._log(m)
            for stmt in (
                f"DROP TABLE IF EXISTS {result_table};",
                f"DROP VIEW IF EXISTS {result_table};",
            ):
                try:
                    agent.db_api.execute_query(agent.user_id, agent.chat_id, stmt)
                except Exception:
                    pass
            return SkillResult(f"Probe result:\n{summary}")
        except Exception as e:
            return SkillResult(f"Error in probe_table: {e}")
