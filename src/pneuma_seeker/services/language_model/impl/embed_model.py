from collections.abc import Generator
from logging import Logger

from numpy import ndarray
from sentence_transformers import SentenceTransformer

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import EmbeddingModelOption, LLMOption
from pneuma_seeker.shared.config import Config


class EmbeddingModel(AbstractModel):
    def __init__(self, config: Config, logger: Logger):
        """
        Designed with "BAAI/bge-base-en-v1.5" in mind, loaded using SentenceTransformers.
        """
        self.model = None
        self.config = config
        self.logger = logger

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.config.EMBED_MODEL_PATH)

    def load_tokenizer(self):
        # No need to load tokenizer
        pass

    def chat(
        self, messages: list[LLMMessage], llm_option: LLMOption | None = None
    ) -> Generator[str, None, None]:
        raise NotImplementedError("Embedding model does not support chat.")

    def batch_chat(self, batch_messages, llm_option=None):
        raise NotImplementedError("Embedding model does not support batch chat.")

    def encode(
        self,
        texts: list[str],
        embed_model_option: EmbeddingModelOption | None = None,
    ) -> ndarray:
        """Embed texts."""
        self.load_model()

        if not embed_model_option:
            embed_model_option = EmbeddingModelOption()

        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(  # type: ignore
            texts,
            batch_size=embed_model_option.batch_size,
            show_progress_bar=False,
        )
