# src/pneuma_seeker/core/materializer/state.py
from pandas import DataFrame

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument


class MaterializerState:
    def __init__(self, user_id: str, chat_id: str, db_api: DBAPI) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.db_api = db_api
        self.T: dict[str, DataFrame] = {}
        self.column_descriptions: dict[str, dict[str, str]] = {}
        self.S: str = ""

        self.retrieved_tables: list[AbstractDocument] = []
        self.external_tables: list[AbstractDocument] = []
        self.intermediate_tables: set[AbstractDocument] = set()
        self.web_search_result: AbstractDocument | None = None
        self.web_crawl_result: AbstractDocument | None = None
        self.join_paths: str | None = None

    def add_intermediate_table(self, table: AbstractDocument):
        if table in self.intermediate_tables:
            self.intermediate_tables.remove(table)
        self.intermediate_tables.add(table)

    def reset(self) -> None:
        self.T = {}
        self.column_descriptions = {}
        self.S = ""

        self.retrieved_tables = []
        self.external_tables = []
        self.intermediate_tables = set()
        self.web_search_result = None
        self.web_crawl_result = None
        self.join_paths = None
