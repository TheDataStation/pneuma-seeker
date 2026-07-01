from dataclasses import dataclass
from enum import Enum


class ConductorResponseType(Enum):
    LOG = "log"
    FINAL_RESPONSE = "final_response"
    DONE = "done"


@dataclass
class ConductorResponse:
    type: ConductorResponseType
    message: str
