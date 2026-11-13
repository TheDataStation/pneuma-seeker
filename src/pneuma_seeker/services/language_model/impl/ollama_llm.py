import json
from collections.abc import Generator
from logging import Logger
from time import time
from typing import Optional

import requests
from numpy import ndarray

from pneuma_seeker.services.language_model.abstract_model import AbstractModel
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import (
    EmbeddingModelOption,
    LLMOption,
)


class OllamaLLM(AbstractModel):
    def __init__(
        self,
        config: Config,
        logger: Logger,
    ):
        self.config = config
        self.logger = logger
        self.base_url = "http://localhost:11434"
        self.total_llm_time = 0.0

    def load_model(self):
        # Ollama loads models lazily
        pass

    def load_tokenizer(self):
        # Not exposed by Ollama
        pass

    def chat(
        self,
        messages: list[LLMMessage],
        llm_option: Optional[LLMOption] = None,
    ) -> Generator[str, None, None]:
        start_llm_time = time()

        stream = False
        temperature = self.config.LLM_TEMPERATURE
        max_tokens = None

        if llm_option:
            stream = llm_option.stream
            if llm_option.temperature is not None:
                temperature = llm_option.temperature
            max_tokens = llm_option.max_new_tokens

        payload = {
            "model": "qwen3.5",
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": {
                **({"temperature": temperature} if temperature is not None else {}),
                **({"num_predict": max_tokens} if max_tokens else {}),
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=stream,
        )

        if stream:
            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)

                message = data.get("message", {})
                content = message.get("content", "")
                thinking = message.get("thinking", "")

                if content:
                    yield content

                if data.get("done"):
                    break

        else:
            data = response.json()
            message = data.get("message", {})
            yield message.get("content", "")

        end_llm_time = time()
        self.total_llm_time += end_llm_time - start_llm_time

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
        raise NotImplementedError("Use Ollama embedding endpoint instead.")
