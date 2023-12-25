from abc import ABC, abstractmethod
from typing import Any


class BaseTextCompletionMixin(ABC):
    @abstractmethod
    def _get_chat_completion(self, **kwargs):
        """Gets a chat completion from the LLM"""
        pass

    @abstractmethod
    def _convert_completion_to_str(self, completion: Any):
        """Converts a completion object to a string"""
        pass


class BaseClassifierMixin(BaseTextCompletionMixin):
    @abstractmethod
    def _extract_out_label(self, completion: Any) -> str:
        """Extracts the label from a completion"""
        pass


class BaseEmbeddingMixin(ABC):
    @abstractmethod
    def _get_embeddings(self, **kwargs):
        """Gets embeddings from the LLM"""
        pass


class BaseTunableMixin(ABC):
    @abstractmethod
    def _tune(self, X: Any, y: Any):
        pass

    @abstractmethod
    def _set_hyperparameters(self, **kwargs):
        pass
