from collections.abc import Generator
from logging import Logger
from time import time
from typing import Optional

from numpy import ndarray
from openai import Omit, OpenAI

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)


class OpenAILLM(AbstractModel):
    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

        self.total_llm_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def load_model(self):
        # OpenAI API does not require model loading
        pass

    def load_tokenizer(self):
        # Tokenizer is handled internally by the API
        pass

    def chat(
        self, messages: list[LLMMessage], llm_option: Optional[LLMOption] = None
    ) -> Generator[str, None, None]:
        start_llm_time = time()
        max_completion_tokens = None
        json_mode = False
        stream = False
        temperature = Omit() if self.config.LLM_PATH.startswith("o") else 0
        if llm_option:
            max_completion_tokens = llm_option.max_new_tokens
            json_mode = llm_option.json_mode
            stream = llm_option.stream
            if llm_option.temperature is not None:
                temperature = llm_option.temperature

        if stream:
            # Stream response as generator of chunks
            response_stream = self.client.chat.completions.create(
                messages=messages,  # type: ignore
                model=self.config.LLM_PATH,
                seed=42,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"} if json_mode else Omit(),
                stream=stream,
                stream_options={"include_usage": True},
            )  # type: ignore
            for event in response_stream:
                self.__accumulate_usage(getattr(event, "usage", None))
                # Some stream events may be keep-alives with empty choices — skip them
                if not getattr(event, "choices", None):
                    continue
                first = event.choices[0]
                delta = getattr(first, "delta", None)
                if delta and getattr(delta, "content", None):
                    chunk = delta.content
                    yield chunk
            end_llm_time = time()
        else:
            # Non-streaming
            response_obj = self.client.chat.completions.create(
                messages=messages,  # type: ignore
                model=self.config.LLM_PATH,
                seed=42,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"} if json_mode else Omit(),
            )
            end_llm_time = time()
            self.__accumulate_usage(getattr(response_obj, "usage", None))
            response = response_obj.choices[0].message.content or ""
            yield response

        self.total_llm_time += end_llm_time - start_llm_time

    def __accumulate_usage(self, usage):
        if usage is not None:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
    
    def reset_metrics(self):
        self.total_llm_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def batch_chat(
        self,
        batch_messages: list[list[LLMMessage]],
        llm_option: Optional[LLMOption] = None,
    ) -> tuple[list[str], int]:
        """
        Chats (in batch) with the model.
        """
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
        """Embed texts."""
        raise NotImplementedError("OpenAI LLM does not support embedding texts.")
