from logging import Logger

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.ir_system.prompt_factory import PromptFactory
from pneuma_seeker.services.core.ir_system.retriever.retriever_factory import (
    RetrieverFactory,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument, RetrieverType


class Retriever:
    """
    Information Retrieval System that manages multiple retrievers
    and handles document indexing and retrieval.
    """

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api

        self.prompt_factory = PromptFactory()
        self.retriever_factory = RetrieverFactory(
            self.user_id, self.chat_id, config, db_api, language_model_api
        )

    def index_documents(
        self, retriever_type: RetrieverType, documents: list[AbstractDocument]
    ):
        """
        Indexes documents into a retriever.
        """
        self.__log(f"Indexing documents on the retriever {retriever_type}.")
        self.retriever_factory.get_retriever(retriever_type).index(documents)
        self.__log("Indexing process is done.")

    def retrieve_documents(
        self,
        retriever_type: RetrieverType,
        prompt: str,
        k: int = 10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves documents from the specified retriever.

        - prompt (str): The query to be given to the retriever.
        - sample_only (bool): Whether to retrieve only a sample of documents.
        - sample_size (int | None): The number of documents to retrieve if sample_only is True.
        """
        retriever = self.retriever_factory.get_retriever(retriever_type)
        documents = retriever.retrieve(prompt, k, sample_only, sample_size)
        return documents

    def retrieve_multi_topic_documents(
        self,
        retriever_type: RetrieverType,
        prompts: list[str],
        k: int = 10,
        sample_only: bool = False,
        sample_size: int | None = None,
    ) -> dict[str, list[AbstractDocument]]:
        """
        Retrieves documents from the specified retriever on multiple topics.

        - prompts (list[str]): The queries corresponding to different topics to be given to the retriever.
        - sample_only (bool): Whether to retrieve only a sample of documents.
        - sample_size (int | None): The number of documents to retrieve if sample_only is True.
        """
        retriever = self.retriever_factory.get_retriever(retriever_type)
        output: dict[str, list[AbstractDocument]] = {}
        for prompt in prompts:
            documents = retriever.retrieve(prompt, k, sample_only, sample_size)
            output[prompt] = documents
        return output

    def __log(self, text: str):
        formatted_log(self.logger, "IR System", text)
