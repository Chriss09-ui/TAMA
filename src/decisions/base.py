"""Decision provider interface: typed probabilistic decisions for agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from decisions.models import DecisionAnswer, DecisionQuestion

State = Union[str, Dict[str, Any], List[Any]]


class DecisionProviderError(RuntimeError):
    """Raised for decision-provider failures.

    `permanent=True` marks configuration/contract errors that must not be
    silently absorbed; agents re-raise these instead of falling back.
    """

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


class DecisionProvider(ABC):
    """A provider of typed decisions: one state, many questions per request."""

    name: str = "decision-provider"
    model: Optional[str] = None

    @abstractmethod
    def ask(
        self,
        state: State,
        questions: Dict[str, DecisionQuestion],
    ) -> Dict[str, DecisionAnswer]:
        """Answer every question about the state, keyed as requested."""
