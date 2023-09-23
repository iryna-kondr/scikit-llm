from abc import ABC, abstractmethod
from typing import Any


class BaseTextCompletionMixin(ABC):
    @abstractmethod
    def _get_chat_completion(self, **kwargs):
        """Gets a chat completion from the LLM"""
        pass


class BaseClassifierMixin(BaseTextCompletionMixin):
    @abstractmethod
    def _extract_out_label(self, completion: Any, **kwargs):
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
