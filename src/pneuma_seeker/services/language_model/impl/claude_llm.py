from collections.abc import Generator
from logging import Logger
from time import time
from typing import Optional

from anthropic import Anthropic
from numpy import ndarray

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)


class ClaudeLLM(AbstractModel):
    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger
        self.model = config.LLM_PATH
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

        self.total_llm_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def load_model(self):
        # Claude API does not require model loading
        pass

    def load_tokenizer(self):
        # Tokenizer is handled internally by the API
        pass

    def chat(
        self, messages: list[LLMMessage], llm_option: Optional[LLMOption] = None
    ) -> Generator[str, None, None]:
        start_llm_time = time()
        max_tokens = 7000
        temperature = None

        if llm_option:
            if llm_option.max_new_tokens is not None:
                max_tokens = llm_option.max_new_tokens
            if llm_option.temperature is not None:
                temperature = llm_option.temperature

        converted_messages: list[dict[str, str]] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            if role not in {"user", "assistant"}:
                role = "user"
            converted_messages.append({"role": role, "content": content})

        request_kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": converted_messages,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        response_obj = self.client.messages.create(**request_kwargs) # type: ignore
        end_llm_time = time()
        self.__accumulate_usage(getattr(response_obj, "usage", None))

        response_text = "".join(
            block.text for block in response_obj.content if hasattr(block, "text")
        )
        yield response_text

        self.total_llm_time += end_llm_time - start_llm_time

    def __accumulate_usage(self, usage):
        if usage is not None:
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens

    def reset_metrics(self):
        self.total_llm_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def batch_chat(
        self,
        batch_messages: list[list[LLMMessage]],
        llm_option: Optional[LLMOption] = None,
    ) -> tuple[list[str], int]:
        responses: list[str] = []

        for messages in batch_messages:
            response = "".join(self.chat(messages, llm_option))
            responses.append(response)

        batch_size = (
            llm_option.batch_size if llm_option and llm_option.batch_size else 1
        )
        return responses, batch_size

    def encode(
        self,
        texts: list[str],
        embed_model_option: EmbeddingModelOption | None = None,
    ) -> ndarray:
        raise NotImplementedError("Claude LLM does not support embedding texts.")
