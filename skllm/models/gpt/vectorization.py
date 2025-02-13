from skllm.models._base.vectorizer import BaseVectorizer as _BaseVectorizer
from skllm.llm.gpt.mixin import GPTEmbeddingMixin as _GPTEmbeddingMixin
from typing import Optional
from model_constants import OPENAI_EMBEDDING_MODEL


class GPTVectorizer(_BaseVectorizer, _GPTEmbeddingMixin):
    def __init__(
        self,
        model: str = OPENAI_EMBEDDING_MODEL,
        batch_size: int = 1,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        """
        Text vectorizer using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use.
        batch_size : int, optional
            number of samples per request, by default 1
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        """
        super().__init__(model=model, batch_size=batch_size)
        self._set_keys(key, org)
