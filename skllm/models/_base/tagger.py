from typing import Any, Union, List, Optional, Dict
from abc import abstractmethod, ABC
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import pandas as pd
from skllm.utils import to_numpy as _to_numpy
from sklearn.base import (
    BaseEstimator as _SklBaseEstimator,
    TransformerMixin as _SklTransformerMixin,
)

from skllm.utils.rendering import display_ner
from skllm.utils.xml import filter_xml_tags, filter_unwanted_entities, json_to_xml
from skllm.prompts.builders import build_ner_prompt
from skllm.prompts.templates import (
    NER_SYSTEM_MESSAGE_TEMPLATE,
    NER_SYSTEM_MESSAGE_SPARSE,
    EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE,
    EXPLAINABLE_NER_SPARSE_PROMPT_TEMPLATE,
)
from skllm.utils import re_naive_json_extractor
import json
from concurrent.futures import ThreadPoolExecutor

class BaseTagger(ABC, _SklBaseEstimator, _SklTransformerMixin):

    num_workers = 1

    def fit(self, X: Any, y: Any = None):
        """
        Fits the model to the data. Usually a no-op.

        Parameters
        ----------
        X : Any
            training data
        y : Any
            training outputs

        Returns
        -------
        self
            BaseTagger
        """
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        return self.transform(X)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Optional[Union[np.ndarray, pd.Series, List[str]]] = None,
    ) -> ndarray:
        return self.fit(X, y).transform(X)

    def transform(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """
        Transforms the input data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.

        Returns
        -------
        List[str]
        """
        X = _to_numpy(X)
        predictions = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            predictions = list(tqdm(executor.map(self._predict_single, X), total=len(X)))
        return np.asarray(predictions)

    def _predict_single(self, x: Any) -> Any:
        prompt_dict = self._get_prompt(x)
        # this will be inherited from the LLM
        prediction = self._get_chat_completion(model=self.model, **prompt_dict)
        prediction = self._convert_completion_to_str(prediction)
        return prediction

    @abstractmethod
    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        pass


class ExplainableNER(BaseTagger):
    entities: Optional[Dict[str, str]] = None
    sparse_output = True
    _allowed_tags = ["entity", "not_entity"]

    display_predictions = False

    def fit(self, X: Any, y: Any = None):
        entities = []
        for k, v in self.entities.items():
            entities.append({"entity": k, "definition": v})
        self.expanded_entities_ = entities
        return self

    def transform(self, X: ndarray | pd.Series | List[str]):
        predictions = super().transform(X)
        if self.sparse_output:
            json_predictions = [
                re_naive_json_extractor(p, expected_output="array") for p in predictions
            ]
            predictions = []
            attributes = ["reasoning", "tag", "value"]
            for x, p in zip(X, json_predictions):
                p_json = json.loads(p)
                predictions.append(
                    json_to_xml(
                        x,
                        p_json,
                        "entity",
                        "not_entity",
                        value_key="value",
                        attributes=attributes,
                    )
                )

        predictions = [
            filter_unwanted_entities(
                filter_xml_tags(p, self._allowed_tags), self.entities
            )
            for p in predictions
        ]
        if self.display_predictions:
            print("Displaying predictions...")
            display_ner(predictions, self.entities)
        return predictions

    def _get_prompt(self, x: str) -> dict:
        if not hasattr(self, "expanded_entities_"):
            raise ValueError("Model not fitted.")
        system_message = (
            NER_SYSTEM_MESSAGE_TEMPLATE.format(entities=self.entities.keys())
            if self.sparse_output
            else NER_SYSTEM_MESSAGE_SPARSE
        )
        template = (
            EXPLAINABLE_NER_SPARSE_PROMPT_TEMPLATE
            if self.sparse_output
            else EXPLAINABLE_NER_DENSE_PROMPT_TEMPLATE
        )
        return {
            "messages": build_ner_prompt(self.expanded_entities_, x, template=template),
            "system_message": system_message.format(entities=self.entities.keys()),
        }
