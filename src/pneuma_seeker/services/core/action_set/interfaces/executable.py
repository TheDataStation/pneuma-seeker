from abc import ABC, abstractmethod
from typing import Any


class Executable(ABC):
    @abstractmethod
    def execute(self, input: dict[str, Any]) -> Any:
        """Executes the action with the given input and returns the output."""
        pass
