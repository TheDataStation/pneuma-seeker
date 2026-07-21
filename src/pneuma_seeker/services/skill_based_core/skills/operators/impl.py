from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult


class ProjectTableSkill(SkillBase):
    name = "project_table"

    def execute(self, args: dict, agent) -> SkillResult:
        src = args.get("src_table_id", "")
        target = args.get("target_table_id", "")
        col_map = args.get("column_mapping", {})
        if not src or not target or not isinstance(col_map, dict):
            return SkillResult(
                "Error: 'src_table_id', 'target_table_id', and 'column_mapping' (dict) are required."
            )
        try:
            df = agent.action_set.project_table(
                src, target, col_map, agent.dataset_name
            )
            if target not in agent.workspace_table_ids:
                agent.workspace_table_ids.append(target)
            return SkillResult(
                f"Created '{target}': {len(df)} row(s), columns: {list(df.columns)}."
            )
        except Exception as e:
            return SkillResult(f"Error in project_table: {e}")


class JoinTablesSkill(SkillBase):
    name = "join_tables"

    def execute(self, args: dict, agent) -> SkillResult:
        left = args.get("left_table_id", "")
        right = args.get("right_table_id", "")
        left_keys = args.get("left_keys", [])
        right_keys = args.get("right_keys", [])
        output = args.get("output_table_id", "")
        if not left or not right or not output:
            return SkillResult(
                "Error: 'left_table_id', 'right_table_id', and 'output_table_id' are required."
            )
        try:
            df = agent.action_set.join_equality(
                left, right, left_keys, right_keys, output, agent.dataset_name
            )
            if output not in agent.workspace_table_ids:
                agent.workspace_table_ids.append(output)
            return SkillResult(
                f"Created '{output}': {len(df)} row(s), columns: {list(df.columns)}."
            )
        except Exception as e:
            return SkillResult(f"Error in join_tables: {e}")


class UnionTablesSkill(SkillBase):
    name = "union_tables"

    def execute(self, args: dict, agent) -> SkillResult:
        table_ids = args.get("table_ids", [])
        output = args.get("output_table_id", "")
        prov_col = args.get("provenance_column", "_source")
        prov_regex = args.get("provenance_regex", "(.*)")
        if not table_ids or not output:
            return SkillResult("Error: 'table_ids' and 'output_table_id' are required.")
        try:
            df = agent.action_set.union_tables(
                table_ids, output, prov_col, prov_regex, agent.dataset_name
            )
            if output not in agent.workspace_table_ids:
                agent.workspace_table_ids.append(output)
            return SkillResult(
                f"Created '{output}': {len(df)} row(s), columns: {list(df.columns)}."
            )
        except Exception as e:
            return SkillResult(f"Error in union_tables: {e}")
