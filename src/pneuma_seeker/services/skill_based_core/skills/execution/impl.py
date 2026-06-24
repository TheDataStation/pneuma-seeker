from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult


class RunSQLSkill(SkillBase):
    name = "run_sql"

    def execute(self, args: dict, agent) -> SkillResult:
        query = args.get("query", "")
        result_id = args.get("result_table_id", "sql_result")
        if not query:
            return SkillResult("Error: 'query' is required.")
        try:
            df = agent.action_set.execute_query(query, result_id)
            if result_id not in agent.workspace_table_ids:
                agent.workspace_table_ids.append(result_id)
            preview = (
                df.head(10).to_string(index=False) if len(df) > 0 else "(empty)"
            )
            return SkillResult(
                f"'{result_id}' created ({len(df)} row(s)).\nPreview:\n{preview}"
            )
        except Exception as e:
            return SkillResult(f"Error in run_sql: {e}")


class RunPythonSkill(SkillBase):
    name = "run_python"

    def execute(self, args: dict, agent) -> SkillResult:
        code = args.get("code", "")
        result_id = args.get("result_table_id", "python_result")
        if not code:
            return SkillResult("Error: 'code' is required.")
        try:
            df = agent.action_set.execute_code(code, result_id)
            if result_id not in agent.workspace_table_ids:
                agent.workspace_table_ids.append(result_id)
            preview = (
                df.head(10).to_string(index=False) if len(df) > 0 else "(empty)"
            )
            return SkillResult(
                f"'{result_id}' created ({len(df)} row(s)).\nPreview:\n{preview}"
            )
        except Exception as e:
            return SkillResult(f"Error in run_python: {e}")
