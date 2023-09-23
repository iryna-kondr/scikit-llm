from skllm.llm.gpt.mixin import (
    GPTClassifierMixin as _GPTClassifierMixin,
    GPTTunableMixin as _GPTTunableMixin,
)
from skllm.models._base.classifier import (
    BaseTunableClassifier as _BaseTunableClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from typing import Optional


class _Tunable(_BaseTunableClassifier, _GPTClassifierMixin, _GPTTunableMixin):
    def _set_hyperparameters(self, base_model: str, n_epochs: int, custom_suffix: str):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.custom_suffix = custom_suffix
        if base_model not in self._supported_tunable_models:
            raise ValueError(
                f"Model {base_model} is not supported. Supported models are"
                f" {self._supported_tunable_models}"
            )


class GPTClassifier(_Tunable, _SingleLabelMixin):
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        default_label: Optional[str] = "Random",
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
        prompt_template: Optional[str] = None,
    ):
        super().__init__(
            model=None, default_label=default_label, prompt_template=prompt_template
        )
        self._set_keys(key, org)
        self._set_hyperparameters(base_model, n_epochs, custom_suffix)


class MultiLabelGPTClassifier(_Tunable, _MultiLabelMixin):
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo-0613",
        default_label: Optional[str] = "Random",
        key: Optional[str] = None,
        org: Optional[str] = None,
        n_epochs: Optional[int] = None,
        custom_suffix: Optional[str] = "skllm",
        prompt_template: Optional[str] = None,
        max_labels: Optional[int] = 5,
    ):
        super().__init__(
            model=None,
            default_label=default_label,
            prompt_template=prompt_template,
            max_labels=max_labels,
        )
        self._set_keys(key, org)
        self._set_hyperparameters(base_model, n_epochs, custom_suffix)
