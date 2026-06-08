from typing import Type

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.services.language_model.impl.azure_openai_embed_model import (
    AzureOpenAIEmbedModel,
)
from pneuma_seeker.services.language_model.impl.azure_openai_llm import AzureOpenAILLM
from pneuma_seeker.services.language_model.impl.embed_model import EmbeddingModel
from pneuma_seeker.services.language_model.impl.claude_llm import ClaudeLLM
from pneuma_seeker.services.language_model.impl.gemini_llm import GeminiLLM
from pneuma_seeker.services.language_model.impl.mock_embed_model import MockEmbedModel
from pneuma_seeker.services.language_model.impl.mock_llm import MockLLM
from pneuma_seeker.services.language_model.impl.ollama_embed_model import OllamaEmbedModel
from pneuma_seeker.services.language_model.impl.ollama_llm import OllamaLLM
from pneuma_seeker.services.language_model.impl.openai_embed_model import (
    OpenAIEmbedModel,
)
from pneuma_seeker.services.language_model.impl.openai_llm import OpenAILLM
from pneuma_seeker.shared.config import Config


def get_llm(config: Config) -> Type[AbstractModel]:
    """Factory function to return the correct LLM instance."""
    normalized_model_path = config.LLM_PATH.lower()
    if (
        "gpt" in normalized_model_path
        or "o3" in normalized_model_path
        or "o4" in normalized_model_path
    ):
        if config.USE_AZURE_LLM:
            return AzureOpenAILLM
        return OpenAILLM
    elif "gemini" in normalized_model_path:
        return GeminiLLM
    elif "claude" in normalized_model_path:
        return ClaudeLLM
    elif "mock" in normalized_model_path:
        return MockLLM
    elif "ollama" in normalized_model_path:
        return OllamaLLM
    else:
        raise ValueError(
            f"No interface implementation for this model path: {config.LLM_PATH}"
        )


def get_embed_model(config: Config) -> Type[AbstractModel]:
    """Factory function to return the correct embedding model class."""
    normalized_model_path = config.EMBED_MODEL_PATH.lower()
    if "text-embedding-3-small" in normalized_model_path:
        if config.USE_AZURE_EMBED_MODEL:
            return AzureOpenAIEmbedModel
        return OpenAIEmbedModel
    elif "mock" in normalized_model_path:
        return MockEmbedModel
    elif "ollama" in normalized_model_path:
        return OllamaEmbedModel
    return EmbeddingModel
