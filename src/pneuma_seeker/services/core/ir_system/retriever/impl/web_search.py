from openai import OpenAI
from pneuma_seeker.shared.schemas.core.ir_system import RetrieverType, Text, AbstractDocument
from pneuma_seeker.services.core.ir_system.retriever.interface import AbstractRetriever


class WebSearch(AbstractRetriever):
    """Represents a web search interface."""

    def __init__(self, user_id, chat_id, config, db_api, language_model_api):
        # Note: For now, we assume OpenAI model
        super().__init__(user_id, chat_id, config, db_api, language_model_api)
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    @property
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        return RetrieverType.WEB_SEARCH

    def load(self):
        """
        Loads the retriever, including its dependencies (e.g., its model).
        """
        pass

    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Retrieves a list of documents given a query.
        """
        response = self.client.responses.create(
            model="o4-mini", tools=[{"type": "web_search"}], input=query
        )
        return [Text("web_search", RetrieverType.WEB_SEARCH, response.output_text, {})]

    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever.
        """
        pass
