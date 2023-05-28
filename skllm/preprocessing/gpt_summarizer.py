from typing import Optional

from skllm.openai.base_gpt import BaseZeroShotGPTTransformer as _BaseGPT
from skllm.prompts.builders import build_summary_prompt


class GPTSummarizer(_BaseGPT):
    system_msg = "You are a text summarizer."
    default_output = "Summary is unavailable."

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        max_words: int = 15,
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model
        self.max_words = max_words

    def _get_prompt(self, X) -> str:
        return build_summary_prompt(X, self.max_words)
