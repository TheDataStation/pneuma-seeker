from collections.abc import Generator
from logging import Logger
from time import time
from typing import Optional

from numpy import ndarray
from openai import OpenAI

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)


class GeminiLLM(AbstractModel):

    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger

        self.client = OpenAI(
            api_key=self.config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        )

        self.total_llm_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def load_model(self):
        pass

    def load_tokenizer(self):
        pass

    def chat(
        self, messages: list[LLMMessage], llm_option: Optional[LLMOption] = None
    ) -> Generator[str, None, None]:
        start_llm_time = time()
        max_completion_tokens = None
        if llm_option:
            max_completion_tokens = llm_option.max_new_tokens
        
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]

        # Non-streaming
        response_obj = self.client.chat.completions.create(
            model=self.config.LLM_PATH,
            messages=formatted_messages,  # type: ignore
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

        responses = []

        for messages in batch_messages:
            response = "".join(
                self.chat(messages, llm_option)
            )
            responses.append(response)

        batch_size = (
            llm_option.batch_size
            if llm_option and llm_option.batch_size
            else 1
        )

        return responses, batch_size

    def encode(
        self,
        texts: list[str],
        embed_model_option: EmbeddingModelOption | None = None,
    ) -> ndarray:
        raise NotImplementedError(
            "Gemini LLM does not support embedding texts."
        )
