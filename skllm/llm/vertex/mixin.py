from typing import Optional, Union, List, Any, Dict, Mapping
from skllm.config import SKLLMConfig as _Config
from skllm.llm.base import (
    BaseClassifierMixin,
    BaseEmbeddingMixin,
    BaseTextCompletionMixin,
    BaseTunableMixin,
)
from skllm.llm.vertex.completion import get_completion_chat_mode, get_completion
from skllm.utils import extract_json_key
import numpy as np
from tqdm import tqdm
import json


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
        else:
            completion = get_completion(model, messages)
        return str(completion)


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
        print(completion)
        try:
            label = extract_json_key(str(completion), "label")
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""
        print(label)
        return label


class VertexEmbeddingMixin(BaseEmbeddingMixin):
    def _get_embeddings(self, text: np.ndarray) -> List[List[float]]:
        raise NotImplementedError("Embeddings are not yet supported for Vertex AI.")


class VertexTunableMixin(BaseTunableMixin):
    # TODO
    def _tune(self, X: Any, y: Any):
        raise NotImplementedError("Tuning is not yet supported for Vertex AI.")
