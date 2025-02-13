from skllm.models._base.text2text import BaseSummarizer as _BaseSummarizer
from skllm.llm.anthropic.mixin import ClaudeTextCompletionMixin as _ClaudeTextCompletionMixin
from typing import Optional
from model_constants import ANTHROPIC_CLAUDE_MODEL


class ClaudeSummarizer(_BaseSummarizer, _ClaudeTextCompletionMixin):
    """Text summarizer using Anthropic Claude API."""
    
    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        key: Optional[str] = None,
        max_words: int = 15,
        focus: Optional[str] = None,
    ) -> None:
        """
        Initialize the Claude summarizer.

        Parameters
        ----------
        model : str, optional
            Model to use
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from global config
        max_words : int, optional
            Soft limit of the summary length, by default 15
        focus : Optional[str], optional
            Concept in the text to focus on, by default None
        """
        self._set_keys(key)
        self.model = model
        self.max_words = max_words
        self.focus = focus
        self.system_message = "You are a text summarizer. Provide concise and accurate summaries."