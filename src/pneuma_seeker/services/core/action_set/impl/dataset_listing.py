from typing import Any

from pneuma_seeker.services.core.action_set.interfaces import Action, Applicable
from pneuma_seeker.services.core.dataset_context import MULTI_DATASET_MODE
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType

# For a dataset nobody has described, a few table names stand in for a description.
_PREVIEW_TABLES = 5


class DatasetListing(Action, Applicable):
    """
    Lists the datasets the user may access: names, sizes and descriptions, without
    searching tables (Table Retrieve does that, with `datasets`). Available only in
    multi-dataset mode (ENABLE_MULTI_DATASET and the per-request `multi_dataset`).
    """

    action_name = ActionNames.DATASET_LISTING
    agents = frozenset({AgentType.CONDUCTOR})
    flag = "ENABLE_MULTI_DATASET"
    mode = MULTI_DATASET_MODE
    order = 0
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.DATASET_LISTING.value}**:\n"
            "    List the datasets you can retrieve tables from.\n"
            "    - **Args**: {}\n"
            "    - **Notes**:\n"
            "        - Returns each dataset's name, table count and description (or, if it has none, a few example table names); it does not search for tables.\n"
            f"        - Then pass the dataset(s) likely to hold the data as `datasets` to `{ActionNames.TABLE_RETRIEVE.value}`. Tables from several datasets can be combined.\n"
            "        - Call this when you don't know which dataset holds the data, or when no retrieved table fits an information need — "
            "do not conclude the data is unavailable before considering the other datasets."
        )

    def apply(self, input: dict[str, Any]) -> str:
        """
        Lists the accessible datasets.

        Expected input keys:
            datasets  list[str]  — the accessible datasets
        """
        datasets: list[str] = input.get("datasets", [])
        if not datasets:
            return "No datasets are accessible."

        lines = ["Accessible datasets:"]
        for dataset in datasets:
            try:
                tables = sorted(self.db_api.list_dataset_tables(dataset))
            except Exception as exc:
                lines.append(f"- {dataset}: (unavailable: {exc})")
                continue
            size = f"{len(tables)} table{'' if len(tables) == 1 else 's'}"
            description = self._description_of(dataset)
            if description:
                lines.append(f"- {dataset} ({size}): {description}")
                continue
            preview = ", ".join(tables[:_PREVIEW_TABLES])
            more = ", ..." if len(tables) > _PREVIEW_TABLES else ""
            lines.append(f"- {dataset} ({size}; e.g. {preview}{more})")
        return "\n".join(lines)

    def _description_of(self, dataset: str) -> str:
        """Returns the dataset's description on one line ("" if none or unavailable)."""
        try:
            return " ".join((self.db_api.get_dataset_description(dataset) or "").split())
        except Exception:
            return ""
