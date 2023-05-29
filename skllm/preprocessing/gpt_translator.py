from typing import Optional

from skllm.openai.base_gpt import BaseZeroShotGPTTransformer as _BaseGPT
from skllm.prompts.builders import build_translation_prompt


class GPTTranslator(_BaseGPT):
    """A text translator."""

    system_msg = "You are a text translator."
    default_output = "Translation is unavailable."

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        output_language: str = "English",
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model
        self.output_language = output_language

    def _get_prompt(self, X: str) -> str:
        """Generates the prompt for the given input.

        Parameters
        ----------
        X : str
            sample to translate

        Returns
        -------
        str
            translated sample
        """
        return build_translation_prompt(X, self.output_language)
