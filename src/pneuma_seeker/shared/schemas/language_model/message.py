from typing import NotRequired, TypedDict


class LLMMessage(TypedDict):
    role: str
    content: str
    is_plan_proposal: NotRequired[bool]
