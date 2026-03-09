from abc import ABC, abstractmethod
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType


class AbstractRetriever(ABC):
    def __init__(
        self, user_id: str, chat_id: str, config: Config, db_api: DBAPI, language_model_api: LanguageModelAPI
    ):
        """
        Initialize the Retriever class
        """
        self.user_id = user_id
        self.chat_id = chat_id
        self.db_api = db_api
        self.language_model_api = language_model_api
        self.is_loaded = False
        self.config = config

    @property
    @abstractmethod
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Loads the retriever, including its dependencies (e.g., its model).
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves a list of documents given a query from certain sources.
        """
        pass

    @abstractmethod
    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever.
        """
        pass
