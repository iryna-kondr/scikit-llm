from typing import Any, List, Optional, Union
from abc import ABC, abstractmethod
from sklearn.base import (
    BaseEstimator as _SklBaseEstimator,
    ClassifierMixin as _SklClassifierMixin,
)
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
from collections import Counter
from skllm.llm.base import (
    BaseClassifierMixin as _BaseClassifierMixin,
    BaseTunableMixin as _BaseTunableMixin,
)
from skllm.utils import to_numpy as _to_numpy
from skllm.prompts.templates import (
    ZERO_SHOT_CLF_PROMPT_TEMPLATE,
    ZERO_SHOT_MLCLF_PROMPT_TEMPLATE,
    FEW_SHOT_CLF_PROMPT_TEMPLATE,
    FEW_SHOT_MLCLF_PROMPT_TEMPLATE,
    ZERO_SHOT_CLF_SHORT_PROMPT_TEMPLATE,
    ZERO_SHOT_MLCLF_SHORT_PROMPT_TEMPLATE,
)
from skllm.prompts.builders import (
    build_zero_shot_prompt_slc,
    build_zero_shot_prompt_mlc,
    build_few_shot_prompt_slc,
    build_few_shot_prompt_mlc,
)
from skllm.memory.base import IndexConstructor
from skllm.memory._sklearn_nn import SklearnMemoryIndex
from skllm.models._base.vectorizer import BaseVectorizer as _BaseVectorizer
import ast

_TRAINING_SAMPLE_PROMPT_TEMPLATE = """
Sample input:
```{x}```
s
Sample target: {label}
"""


class SingleLabelMixin:
    """Mixin class for single label classification."""

    def validate_prediction(self, label: Any) -> str:
        """
        Validates a prediction.

        Parameters
        ----------
        label : str
            The label to validate.

        Returns
        -------
        str
            The validated label.
        """
        if label not in self.classes_:
            label = str(label).replace("'", "").replace('"', "")
            if label not in self.classes_:  # try again
                label = self._get_default_label()
        return label

    def _extract_labels(self, y: Any) -> List[str]:
        """
        Return the class labels as a list.

        Parameters
        ----------
        y : Any

        Returns
        -------
        List[str]
        """
        if isinstance(y, (pd.Series, np.ndarray)):
            labels = y.tolist()
        else:
            labels = y
        return labels


class MultiLabelMixin:
    """Mixin class for multi label classification."""

    def validate_prediction(self, label: Any) -> List[str]:
        """
        Validates a prediction.

        Parameters
        ----------
        label : Any
            The label to validate.

        Returns
        -------
        List[str]
            The validated label.
        """
        if not isinstance(label, list):
            label = []
        filtered_labels = []
        for l in label:
            if l in self.classes_ and l:
                if l not in filtered_labels:
                    filtered_labels.append(l)
            elif la := l.replace("'", "").replace('"', "") in self.classes_:
                if la not in filtered_labels:
                    filtered_labels.append(la)
            else:
                default_label = self._get_default_label()
                if not (
                    self.default_label == "Random" and default_label in filtered_labels
                ):
                    filtered_labels.append(default_label)
        filtered_labels.extend([""] * self.max_labels)
        return filtered_labels[: self.max_labels]

    def _extract_labels(self, y) -> List[str]:
        """Extracts the labels into a list.

        Parameters
        ----------
        y : Any

        Returns
        -------
        List[str]
        """
        labels = []
        for l in y:
            for j in l:
                labels.append(j)
        return labels


class BaseClassifier(ABC, _SklBaseEstimator, _SklClassifierMixin):
    system_msg = "You are a text classifier."

    def __init__(
        self,
        model: Optional[str],  # model can initially be None for tunable estimators
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(self, _BaseClassifierMixin):
            raise TypeError(
                "Classifier must be mixed with a skllm.llm.base.BaseClassifierMixin"
                " class"
            )
        if not isinstance(self, (SingleLabelMixin, MultiLabelMixin)):
            raise TypeError(
                "Classifier must be mixed with a SingleLabelMixin or MultiLabelMixin"
                " class"
            )

        self.model = model
        if not isinstance(default_label, str):
            raise TypeError("default_label must be a string")
        self.default_label = default_label
        if isinstance(self, MultiLabelMixin):
            if not isinstance(max_labels, int):
                raise TypeError("max_labels must be an integer")
            if max_labels < 2:
                raise ValueError("max_labels must be greater than 1")
            self.max_labels = max_labels
        if prompt_template is not None and not isinstance(prompt_template, str):
            raise TypeError("prompt_template must be a string or None")
        self.prompt_template = prompt_template

    def _predict_single(self, x: Any) -> Any:
        prompt_dict = self._get_prompt(x)
        # this will be inherited from the LLM
        prediction = self._get_chat_completion(model=self.model, **prompt_dict)
        prediction = self._extract_out_label(prediction)
        # this will be inherited from the sl/ml mixin
        prediction = self.validate_prediction(prediction)
        return prediction

    @abstractmethod
    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        pass

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        """
        Fits the model to the given data.

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            Training data
        y : Union[np.ndarray, pd.Series, List[str], List[List[str]]]
            Training labels

        Returns
        -------
        BaseClassifier
            self
        """
        X = _to_numpy(X)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]], num_workers: int = 1):
        """
        Predicts the class of each input.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.
            
        num_workers : int
            number of workers to use for multithreaded prediction, default 1

        Returns
        -------
        np.ndarray
            The predicted classes as a numpy array.
        """
        X = _to_numpy(X)
        
        if num_workers > 1:
            warnings.warn("Passing num_workers to predict is temporary and will be removed in the future.")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            predictions = list(tqdm(executor.map(self._predict_single, X), total=len(X)))
            
        return np.array(predictions)

    def _get_unique_targets(self, y: Any):
        labels = self._extract_labels(y)

        counts = Counter(labels)

        total = sum(counts.values())

        classes, probs = [], []
        for l, c in counts.items():
            classes.append(l)
            probs.append(c / total)

        return classes, probs

    def _get_default_label(self):
        """Returns the default label based on the default_label argument."""
        if self.default_label == "Random":
            return random.choices(self.classes_, self.probabilities_)[0]
        else:
            return self.default_label


class BaseZeroShotClassifier(BaseClassifier):
    def _get_prompt_template(self) -> str:
        """Returns the prompt template to use for a single input."""
        if self.prompt_template is not None:
            return self.prompt_template
        elif isinstance(self, SingleLabelMixin):
            return ZERO_SHOT_CLF_PROMPT_TEMPLATE
        return ZERO_SHOT_MLCLF_PROMPT_TEMPLATE

    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        if isinstance(self, SingleLabelMixin):
            prompt = build_zero_shot_prompt_slc(
                x, repr(self.classes_), template=self._get_prompt_template()
            )
        else:
            prompt = build_zero_shot_prompt_mlc(
                x,
                repr(self.classes_),
                self.max_labels,
                template=self._get_prompt_template(),
            )
        return {"messages": prompt, "system_message": self.system_msg}


class BaseFewShotClassifier(BaseClassifier):
    def _get_prompt_template(self) -> str:
        """Returns the prompt template to use for a single input."""
        if self.prompt_template is not None:
            return self.prompt_template
        elif isinstance(self, SingleLabelMixin):
            return FEW_SHOT_CLF_PROMPT_TEMPLATE
        return FEW_SHOT_MLCLF_PROMPT_TEMPLATE

    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        training_data = []
        for xt, yt in zip(*self.training_data_):
            training_data.append(
                _TRAINING_SAMPLE_PROMPT_TEMPLATE.format(x=xt, label=yt)
            )

        training_data_str = "\n".join(training_data)

        if isinstance(self, SingleLabelMixin):
            prompt = build_few_shot_prompt_slc(
                x,
                repr(self.classes_),
                training_data=training_data_str,
                template=self._get_prompt_template(),
            )
        else:
            prompt = build_few_shot_prompt_mlc(
                x,
                repr(self.classes_),
                training_data=training_data_str,
                max_cats=self.max_labels,
                template=self._get_prompt_template(),
            )
        return {"messages": prompt, "system_message": "You are a text classifier."}

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        """
        Fits the model to the given data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            Training data
        y : Union[np.ndarray, pd.Series, List[str]]
            Training labels

        Returns
        -------
        BaseFewShotClassifier
            self
        """
        if not len(X) == len(y):
            raise ValueError("X and y must have the same length.")
        X = _to_numpy(X)
        y = _to_numpy(y)
        self.training_data_ = (X, y)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)
        return self


class BaseDynamicFewShotClassifier(BaseClassifier):
    def __init__(
        self,
        model: str,
        default_label: str = "Random",
        n_examples=3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: _BaseVectorizer = None,
        prompt_template: Optional[str] = None,
        metric="euclidean",
    ):
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
        )
        self.vectorizer = vectorizer
        self.memory_index = memory_index
        self.n_examples = n_examples
        self.metric = metric
        if isinstance(self, MultiLabelMixin):
            raise TypeError("Multi-label classification is not supported")

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, List[str]],
        y: Union[np.ndarray, pd.Series, List[str]],
    ):
        """
        Fits the model to the given data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            Training data
        y : Union[np.ndarray, pd.Series, List[str]]
            Training labels

        Returns
        -------
        BaseDynamicFewShotClassifier
            self
        """

        if not self.vectorizer:
            raise ValueError("Vectorizer must be set")
        X = _to_numpy(X)
        y = _to_numpy(y)
        self.embedding_model_ = self.vectorizer.fit(X)
        self.classes_, self.probabilities_ = self._get_unique_targets(y)

        self.data_ = {}
        for cls in self.classes_:
            print(f"Building index for class `{cls}` ...")
            self.data_[cls] = {}
            partition = X[y == cls]
            self.data_[cls]["partition"] = partition
            embeddings = self.embedding_model_.transform(partition)
            if self.memory_index is not None:
                index = self.memory_index()
                index.dim = embeddings.shape[1]
            else:
                index = SklearnMemoryIndex(embeddings.shape[1], metric=self.metric)
            for embedding in embeddings:
                index.add(embedding)
            index.build()
            self.data_[cls]["index"] = index

        return self

    def _get_prompt_template(self) -> str:
        """Returns the prompt template to use for a single input."""
        if self.prompt_template is not None:
            return self.prompt_template
        return FEW_SHOT_CLF_PROMPT_TEMPLATE

    def _get_prompt(self, x: str) -> dict:
        """
        Generates the prompt for the given input.

        Parameters
        ----------
        x : str
            sample to classify

        Returns
        -------
        dict
            final prompt
        """
        embedding = self.embedding_model_.transform([x])
        training_data = []
        for cls in self.classes_:
            index = self.data_[cls]["index"]
            partition = self.data_[cls]["partition"]
            neighbors = index.retrieve(embedding, min(self.n_examples, len(partition)))
            neighbors = [partition[i] for i in neighbors[0]]
            training_data.extend(
                [
                    _TRAINING_SAMPLE_PROMPT_TEMPLATE.format(x=neighbor, label=cls)
                    for neighbor in neighbors
                ]
            )

        training_data_str = "\n".join(training_data)

        msg = build_few_shot_prompt_slc(
            x=x,
            training_data=training_data_str,
            labels=repr(self.classes_),
            template=self._get_prompt_template(),
        )

        return {"messages": msg, "system_message": "You are a text classifier."}


class BaseTunableClassifier(BaseClassifier):
    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        """
        Fits the model to the given data.

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            Training data
        y : Union[np.ndarray, pd.Series, List[str], List[List[str]]]
            Training labels

        Returns
        -------
        BaseTunableClassifier
            self
        """
        if not isinstance(self, _BaseTunableMixin):
            raise TypeError(
                "Classifier must be mixed with a skllm.llm.base.BaseTunableMixin class"
            )
        super().fit(X, y)
        self._tune(X, y)
        return self

    def _get_prompt_template(self) -> str:
        """Returns the prompt template to use for a single input."""
        if self.prompt_template is not None:
            return self.prompt_template
        elif isinstance(self, SingleLabelMixin):
            return ZERO_SHOT_CLF_SHORT_PROMPT_TEMPLATE
        return ZERO_SHOT_MLCLF_SHORT_PROMPT_TEMPLATE

    def _get_prompt(self, x: str) -> dict:
        """Returns the prompt to use for a single input."""
        if isinstance(self, SingleLabelMixin):
            prompt = build_zero_shot_prompt_slc(
                x, repr(self.classes_), template=self._get_prompt_template()
            )
        else:
            prompt = build_zero_shot_prompt_mlc(
                x,
                repr(self.classes_),
                self.max_labels,
                template=self._get_prompt_template(),
            )
        return {"messages": prompt, "system_message": "You are a text classifier."}
