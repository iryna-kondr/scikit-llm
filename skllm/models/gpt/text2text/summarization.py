from skllm.models._base.text2text import BaseSummarizer as _BaseSummarizer
from skllm.llm.gpt.mixin import GPTTextCompletionMixin as _GPTTextCompletionMixin
from typing import Optional


class GPTSummarizer(_BaseSummarizer, _GPTTextCompletionMixin):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        key: Optional[str] = None,
        org: Optional[str] = None,
        max_words: int = 15,
        focus: Optional[str] = None,
    ) -> None:
        """
        Text summarizer using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        max_words : int, optional
            soft limit of the summary length, by default 15
        focus : Optional[str], optional
            concept in the text to focus on, by default None
        """
        self._set_keys(key, org)
        self.model = model
        self.max_words = max_words
        self.focus = focus
