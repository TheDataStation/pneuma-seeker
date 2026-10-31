import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from pneuma_seeker.services.language_model import model_factory
from pneuma_seeker.services.language_model.impl.azure_openai_embed_model import (
    AzureOpenAIEmbedModel,
)
from pneuma_seeker.services.language_model.impl.azure_openai_llm import AzureOpenAILLM
from pneuma_seeker.services.language_model.impl.embed_model import EmbeddingModel
from pneuma_seeker.services.language_model.impl.openai_embed_model import (
    OpenAIEmbedModel,
)
from pneuma_seeker.services.language_model.impl.openai_llm import OpenAILLM
from pneuma_seeker.shared.config import Config


class ModelFactoryTests(unittest.TestCase):
    def test_get_llm_returns_openai_by_default(self):
        cfg = Config()
        cfg.LLM_PATH = "gpt-4.1-mini"
        cfg.USE_AZURE_LLM = False
        cls = model_factory.get_llm(cfg)
        self.assertIs(cls, OpenAILLM)

    def test_get_llm_returns_azure_when_use_azure_true(self):
        cfg = Config()
        cfg.LLM_PATH = "o4-something"
        cfg.USE_AZURE_LLM = True
        cls = model_factory.get_llm(cfg)
        self.assertIs(cls, AzureOpenAILLM)

    def test_get_llm_raises_for_unknown(self):
        cfg = Config()
        cfg.LLM_PATH = "some-unsupported-model"
        cfg.USE_AZURE_LLM = False
        with self.assertRaises(ValueError):
            model_factory.get_llm(cfg)

    def test_get_embed_model_returns_azure_embed_when_use_azure_true(self):
        cfg = Config()
        cfg.EMBED_MODEL_PATH = "text-embedding-3-small"
        cfg.USE_AZURE_EMBED_MODEL = True
        cls = model_factory.get_embed_model(cfg)
        self.assertIs(cls, AzureOpenAIEmbedModel)

    def test_get_embed_model_returns_openai_embed_when_use_azure_false(self):
        cfg = Config()
        cfg.EMBED_MODEL_PATH = "text-embedding-3-small"
        cfg.USE_AZURE_EMBED_MODEL = False
        cls = model_factory.get_embed_model(cfg)
        self.assertIs(cls, OpenAIEmbedModel)

    def test_get_embed_model_returns_generic_for_other(self):
        cfg = Config()
        cfg.EMBED_MODEL_PATH = "other-embed"
        cls = model_factory.get_embed_model(cfg)
        self.assertIs(cls, EmbeddingModel)


if __name__ == "__main__":
    unittest.main()
