from skllm.models._base.tagger import ExplainableNER as _ExplainableNER
from skllm.llm.gpt.mixin import GPTTextCompletionMixin as _GPTTextCompletionMixin
from typing import Optional, Dict


class GPTExplainableNER(_ExplainableNER, _GPTTextCompletionMixin):
    def __init__(
        self,
        entities: Dict[str, str],
        display_predictions: bool = False,
        sparse_output: bool = True,
        model: str = "gpt-4o",
        key: Optional[str] = None,
        org: Optional[str] = None,
        num_workers: int = 1,
    ) -> None:
        """
        Named entity recognition using OpenAI/GPT API-compatible models.

        Parameters
        ----------
        entities : dict
            dictionary of entities to recognize, with keys as entity names and values as descriptions
        display_predictions : bool, optional
            whether to display predictions, by default False
        sparse_output : bool, optional
            whether to generate a sparse representation of the predictions, by default True
        model : str, optional
            model to use, by default "gpt-4o"
        key : Optional[str], optional
            estimator-specific API key; if None, retrieved from the global config, by default None
        org : Optional[str], optional
            estimator-specific ORG key; if None, retrieved from the global config, by default None
        num_workers : int, optional
            number of workers (threads) to use, by default 1
        """
        self._set_keys(key, org)
        self.model = model
        self.entities = entities
        self.display_predictions = display_predictions
        self.sparse_output = sparse_output
        self.num_workers = num_workers