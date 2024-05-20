from typing import Optional, Union, List, Any, Dict
from skllm.llm.base import (
    BaseClassifierMixin,
    BaseEmbeddingMixin,
    BaseTextCompletionMixin,
    BaseTunableMixin,
)
from skllm.llm.vertex.tuning import tune
from skllm.llm.vertex.completion import get_completion_chat_mode, get_completion, get_completion_chat_gemini
from skllm.utils import extract_json_key
import numpy as np
import pandas as pd


class VertexMixin:
    pass


class VertexTextCompletionMixin(BaseTextCompletionMixin):
    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
        system_message: Optional[str],
        examples: Optional[List] = None,
    ) -> str:
        if examples is not None:
            raise NotImplementedError(
                "Examples API is not yet supported for Vertex AI."
            )
        if not isinstance(messages, str):
            raise ValueError("Only messages as strings are supported.")
        if model.startswith("chat-"):
            completion = get_completion_chat_mode(model, system_message, messages)
        elif model.startswith("gemini-"):
            completion = get_completion_chat_gemini(model, system_message, messages)
        else:
            completion = get_completion(model, messages)
        return str(completion)

    def _convert_completion_to_str(self, completion: str) -> str:
        return completion


class VertexClassifierMixin(BaseClassifierMixin, VertexTextCompletionMixin):
    def _extract_out_label(self, completion: str, **kwargs) -> Any:
        """Extracts the label from a completion.

        Parameters
        ----------
        label : Mapping[str, Any]
            The label to extract.

        Returns
        -------
        label : str
        """
        try:
            label = extract_json_key(str(completion), "label")
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""
        return label


class VertexEmbeddingMixin(BaseEmbeddingMixin):
    def _get_embeddings(self, text: np.ndarray) -> List[List[float]]:
        raise NotImplementedError("Embeddings are not yet supported for Vertex AI.")


class VertexTunableMixin(BaseTunableMixin):
    _supported_tunable_models = ["text-bison@002"]

    def _set_hyperparameters(self, base_model: str, n_update_steps: int, **kwargs):
        self.verify_model_is_supported(base_model)
        self.base_model = base_model
        self.n_update_steps = n_update_steps

    def verify_model_is_supported(self, model: str):
        if model not in self._supported_tunable_models:
            raise ValueError(
                f"Model {model} is not supported. Supported models are"
                f" {self._supported_tunable_models}"
            )

    def _tune(self, X: Any, y: Any):
        df = pd.DataFrame({"input_text": X, "output_text": y})
        job = tune(self.base_model, df, self.n_update_steps)._job
        tuned_model = job.result()
        self.tuned_model_ = tuned_model._model_resource_name
        self.model = tuned_model._model_resource_name
        return self
