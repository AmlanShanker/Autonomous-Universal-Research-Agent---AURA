from abc import ABC, abstractmethod


class ResearchTool(ABC):
    """
    Base interface for all AURA research tools.

    Every tool must have:
    - a unique name
    - a description
    - an execute method
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """
        Execute the tool and return a structured result.
        """
        pass
