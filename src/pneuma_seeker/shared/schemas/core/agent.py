from enum import Enum


class AgentType(str, Enum):
    CONDUCTOR = "conductor"
    MATERIALIZER = "materializer"
