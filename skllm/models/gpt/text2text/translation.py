from skllm.models._base.text2text import BaseTranslator as _BaseTranslator
from skllm.llm.gpt.mixin import GPTTextCompletionMixin as _GPTTextCompletionMixin
from typing import Optional


class GPTTranslator(_BaseTranslator, _GPTTextCompletionMixin):
    default_output = "Translation is unavailable."

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        key: Optional[str] = None,
        org: Optional[str] = None,
        output_language: str = "English",
    ) -> None:
        """
        Text translator using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        output_language : str, optional
            target language, by default "English"
        """
        self._set_keys(key, org)
        self.model = model
        self.output_language = output_language
