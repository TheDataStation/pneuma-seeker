from collections.abc import Generator
from logging import Logger
from time import time
from typing import Optional

from numpy import ndarray
from openai import AzureOpenAI

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)


class AzureOpenAILLM(AbstractModel):
    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger
        self.client = AzureOpenAI(
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )

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
        temperature = 1
        if llm_option:
            max_completion_tokens = llm_option.max_new_tokens
            json_mode = llm_option.json_mode
            stream = llm_option.stream
            if llm_option.temperature is not None:
                temperature = llm_option.temperature

        if stream:
            # Stream response as generator of chunks
            if json_mode:
                response_stream = self.client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=self.config.LLM_PATH,
                    seed=42,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    response_format={"type": "json_object"},
                    stream=stream,
                    stream_options={"include_usage": True},
                )  # type: ignore
            else:
                response_stream = self.client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=self.config.LLM_PATH,
                    seed=42,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    stream=stream,
                    stream_options={"include_usage": True},
                )  # type: ignore

            for event in response_stream:
                self.__accumulate_usage(getattr(event, "usage", None))
                if not getattr(
                    event, "choices", None
                ):  # skip keep-alives or DONE packets
                    continue
                delta = event.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    chunk = delta.content
                    yield chunk
            end_llm_time = time()
        else:
            # Non-streaming
            if json_mode:
                response_obj = self.client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=self.config.LLM_PATH,
                    seed=42,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                response_obj = self.client.chat.completions.create(
                    messages=messages,  # type: ignore
                    model=self.config.LLM_PATH,
                    seed=42,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
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
        raise NotImplementedError("Azure OpenAI LLM does not support embedding texts.")
