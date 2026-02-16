from enum import Enum


class DocumentType(Enum):
    TARGET_TABLE = "target_table"
    INTERMEDIATE_TABLE = "intermediate_table"
    EXTERNAL_TABLE = "external_table"

    RETRIEVED_TABLE = "retrieved_table"
    ENUMERATED_TABLE = "enumerated_table"

    WEB_SEARCH_RESULT = "web_search_result"
    WEB_CRAWL_RESULT = "web_crawl_result"
