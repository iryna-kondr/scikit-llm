from skllm.models._base.text2text import BaseTranslator as _BaseTranslator
from skllm.llm.anthropic.mixin import ClaudeTextCompletionMixin as _ClaudeTextCompletionMixin
from typing import Optional
from model_constants import ANTHROPIC_CLAUDE_MODEL


class ClaudeTranslator(_BaseTranslator, _ClaudeTextCompletionMixin):
    """Text translator using Anthropic Claude API."""
    
    default_output = "Translation is unavailable."

    def __init__(
        self,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        key: Optional[str] = None,
        output_language: str = "English",
    ) -> None:
        """
        Initialize the Claude translator.

        Parameters
        ----------
        model : str, optional
            Model to use
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from global config
        output_language : str, optional
            Target language, by default "English"
        """
        self._set_keys(key)
        self.model = model
        self.output_language = output_language
        self.system_message = (
            "You are a professional translator. Provide accurate translations "
            "while maintaining the original meaning and tone of the text."
        )