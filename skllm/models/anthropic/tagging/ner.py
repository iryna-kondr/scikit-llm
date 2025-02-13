from skllm.models._base.tagger import ExplainableNER as _ExplainableNER
from skllm.llm.anthropic.mixin import ClaudeTextCompletionMixin as _ClaudeTextCompletionMixin
from typing import Optional, Dict
from model_constants import ANTHROPIC_CLAUDE_MODEL


class AnthropicExplainableNER(_ExplainableNER, _ClaudeTextCompletionMixin):
    """Named Entity Recognition model using Anthropic's Claude API for explainable entity extraction."""

    def __init__(
        self,
        entities: Dict[str, str],
        display_predictions: bool = False,
        sparse_output: bool = True,
        model: str = ANTHROPIC_CLAUDE_MODEL,
        key: Optional[str] = None,
        num_workers: int = 1,
    ) -> None:
        """
        Named entity recognition using Anthropic Claude API.

        Parameters
        ----------
        entities : dict
            dictionary of entities to recognize, with keys as entity names and values as descriptions
        display_predictions : bool, optional
            whether to display predictions, by default False
        sparse_output : bool, optional
            whether to generate a sparse representation of the predictions, by default True
        model : str, optional
            model to use
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config
        num_workers : int, optional
            number of workers (threads) to use, by default 1
        """
        self._set_keys(key)
        self.model = model
        self.entities = entities
        self.display_predictions = display_predictions
        self.sparse_output = sparse_output
        self.num_workers = num_workers