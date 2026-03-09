from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.retriever.interface import (
    AbstractRetriever,
)
from pneuma_seeker.services.core.ir_system.retriever.impl.enumerator import Enumerator
from pneuma_seeker.services.core.ir_system.retriever.impl.pneuma_retriever import (
    PneumaRetriever,
)
from pneuma_seeker.services.core.ir_system.retriever.impl.document_db import DocumentDB
from pneuma_seeker.services.core.ir_system.retriever.impl.web_crawler import WebCrawler
from pneuma_seeker.services.core.ir_system.retriever.impl.web_search import WebSearch
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType


class RetrieverFactory:
    """Factory class to create Retriever instances based on the RetrieverType."""

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ):
        """Initialize the RetrieverFactory with available retriever instances."""
        self.retriever_instances = {
            RetrieverType.PNEUMA_RETRIEVER: PneumaRetriever(
                user_id, chat_id, config, db_api, language_model_api
            ),
            RetrieverType.DOCUMENT_DB: DocumentDB(user_id, chat_id, config, db_api, language_model_api),
            RetrieverType.WEB_SEARCH: WebSearch(user_id, chat_id, config, db_api, language_model_api),
            RetrieverType.ENUMERATOR: Enumerator(user_id, chat_id, config, db_api, language_model_api),
            RetrieverType.WEB_CRAWL: WebCrawler(user_id, chat_id, config, db_api, language_model_api),
        }

    def get_retriever(self, retriever_type: RetrieverType) -> AbstractRetriever:
        """Factory function to return the correct Retriever instance."""
        return self.retriever_instances[retriever_type]
