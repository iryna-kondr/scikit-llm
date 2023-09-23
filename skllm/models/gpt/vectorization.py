from skllm.models._base.vectorizer import BaseVectorizer as _BaseVectorizer
from skllm.llm.gpt.mixin import GPTEmbeddingMixin as _GPTEmbeddingMixin
from typing import Optional


class GPTVectorizer(_BaseVectorizer, _GPTEmbeddingMixin):
    """
    A vectorizer that uses GPT embeddings.
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 1,
        key: Optional[str] = None,
        org: Optional[str] = None,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self._set_keys(key, org)
