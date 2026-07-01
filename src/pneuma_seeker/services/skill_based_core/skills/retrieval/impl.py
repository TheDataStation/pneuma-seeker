from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult
from pneuma_seeker.shared.schemas.core.ir_system import (
    RetrieverType,
    convert_retrieval_results_to_str,
)


class WebSearchSkill(SkillBase):
    name = "web_search"
    config_flag = "ENABLE_WEB_SEARCH"

    def execute(self, args: dict, agent) -> SkillResult:
        query = args.get("query", "")
        if not query:
            return SkillResult("Error: 'query' is required.")
        results = agent.action_set.retrieve_documents(query, RetrieverType.WEB_SEARCH)
        if results:
            agent.web_search_result = results[0]
        return SkillResult(
            f"Web search results for '{query}':\n"
            + convert_retrieval_results_to_str(results)
        )


class WebCrawlSkill(SkillBase):
    name = "web_crawl"
    config_flag = "ENABLE_WEB_CRAWL"

    def execute(self, args: dict, agent) -> SkillResult:
        url = args.get("url", "")
        if not url:
            return SkillResult("Error: 'url' is required.")
        results = agent.action_set.retrieve_documents(url, RetrieverType.WEB_CRAWL)
        if results:
            agent.web_crawl_result = results[0]
        return SkillResult(
            f"Web crawl results for '{url}':\n"
            + convert_retrieval_results_to_str(results)
        )


class RetrieveTablesSkill(SkillBase):
    name = "retrieve_tables"

    def execute(self, args: dict, agent) -> SkillResult:
        prompts = args.get("prompts")
        if not isinstance(prompts, list) or not all(
            isinstance(p, str) for p in prompts
        ):
            return SkillResult("Error: 'prompts' must be a list of strings.")
        agent.retrieved_tables = agent.action_set.retrieve_multi_topic_documents(
            prompts, RetrieverType.PNEUMA_RETRIEVER, 10, True, 3
        )
        join_paths: str | None = None
        try:
            join_paths = agent.action_set.discover_join_paths(agent.retrieved_tables)
        except Exception:
            pass
        parts = [
            f"Retrieved {len(agent.retrieved_tables)} table(s):\n"
            + convert_retrieval_results_to_str(agent.retrieved_tables)
        ]
        if join_paths:
            parts.append(f"Potential join paths:\n{join_paths}")
        return SkillResult("\n".join(parts))


class EnumerateTablesSkill(SkillBase):
    name = "enumerate_tables"

    def execute(self, args: dict, agent) -> SkillResult:
        patterns = args.get("patterns")
        if not isinstance(patterns, list) or not all(
            isinstance(p, str) for p in patterns
        ):
            return SkillResult("Error: 'patterns' must be a list of strings.")
        agent.enumerated_tables = agent.action_set.retrieve_multi_topic_documents(
            patterns, RetrieverType.ENUMERATOR, 20, True, 2
        )
        ids = [t.doc_id for t in agent.enumerated_tables]
        return SkillResult(f"Tables matching {patterns}: {ids}")
