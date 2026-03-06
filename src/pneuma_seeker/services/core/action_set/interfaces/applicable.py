from abc import ABC, abstractmethod
from typing import Any


class Applicable(ABC):
    @abstractmethod
    def apply(self, input: dict[str, Any]) -> Any:
        """Applies the action with the given input and returns the output."""
        pass
