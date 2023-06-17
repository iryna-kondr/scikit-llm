from typing import Any, List, Optional, Union

import numpy as np
from numpy import ndarray
from pandas import Series

from skllm.openai.base_gpt import BaseZeroShotGPTTransformer as _BaseGPT
from skllm.prompts.builders import build_focused_summary_prompt, build_summary_prompt


class GPTSummarizer(_BaseGPT):
    """
    A text summarizer.
    
    Parameters
    ----------
    openai_key : str, optional
        The OPEN AI key to use. Defaults to None.
    openai_org : str, optional
        The OPEN AI organization ID to use. Defaults to None.
    openai_model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    max_words : int, optional
        The maximum number of words to use in the summary. Defaults to 15.
    
    """
    system_msg = "You are a text summarizer."
    default_output = "Summary is unavailable."

    def __init__(
        self,
        openai_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        max_words: int = 15,
        focus: Optional[str] = None,
    ):
        self._set_keys(openai_key, openai_org)
        self.openai_model = openai_model
        self.max_words = max_words
        self.focus = focus

    def _get_prompt(self, X: str) -> str:
        """
        Generates the prompt for the given input.

        Parameters
        ----------
        X : str
            sample to summarize

        Returns
        -------
        str
        """
        if self.focus:
            return build_focused_summary_prompt(X, self.max_words, self.focus)
        else:
            return build_summary_prompt(X, self.max_words)

    def transform(self, X: Union[ndarray, Series, List[str]], **kwargs: Any) -> ndarray:
        y = super().transform(X, **kwargs)
        if self.focus:
            # remove "Mentioned concept is not present in the text." from the output
            y = np.asarray(
                [
                    i.replace("Mentioned concept is not present in the text.", "")
                    .replace("The general summary is:", "")
                    .strip()
                    for i in y
                ],
                dtype=object,
            )
        return y
