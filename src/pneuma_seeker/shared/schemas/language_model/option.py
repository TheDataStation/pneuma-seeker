from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMOption:
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    penalty_alpha: Optional[float] = None
    json_mode: bool = False
    batch_size: Optional[int] = None
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = None


@dataclass
class EmbeddingModelOption:
    batch_size: int = 32
