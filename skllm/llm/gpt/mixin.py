from typing import Optional, Union, List, Any, Dict, Mapping
from skllm.config import SKLLMConfig as _Config
from skllm.llm.gpt.completion import get_chat_completion
from skllm.llm.gpt.embedding import get_embedding
from skllm.llm.base import (
    BaseClassifierMixin,
    BaseEmbeddingMixin,
    BaseTextCompletionMixin,
    BaseTunableMixin,
)
from skllm.llm.gpt.clients.openai.tuning import (
    create_tuning_job,
    await_results,
    delete_file,
)
import uuid
from skllm.llm.gpt.clients.openai.credentials import (
    set_credentials as _set_credentials_openai,
)
from skllm.utils import extract_json_key
import numpy as np
from tqdm import tqdm
import json


def construct_message(role: str, content: str) -> dict:
    """Constructs a message for the OpenAI API.

    Parameters
    ----------
    role : str
        The role of the message. Must be one of "system", "user", or "assistant".
    content : str
        The content of the message.

    Returns
    -------
    message : dict
    """
    if role not in ("system", "user", "assistant"):
        raise ValueError("Invalid role")
    return {"role": role, "content": content}


def _build_clf_example(
    x: str, y: str, system_msg="You are a text classification model."
):
    sample = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": x},
            {"role": "assistant", "content": y},
        ]
    }
    return json.dumps(sample)


class GPTMixin:
    """
    A mixin class that provides OpenAI key and organization to other classes.
    """

    _prefer_json_output = False

    def _set_keys(self, key: Optional[str] = None, org: Optional[str] = None) -> None:
        """
        Set the OpenAI key and organization.
        """
            
        self.key = key
        self.org = org
        

    def _get_openai_key(self) -> str:
        """
        Get the OpenAI key from the class or the config file.

        Returns
        -------
        key: str
        """
        key = self.key
        if key is None:
            key = _Config.get_openai_key()
        if key is None:
            raise RuntimeError("OpenAI key was not found")
        return key

    def _get_openai_org(self) -> str:
        """
        Get the OpenAI organization ID from the class or the config file.

        Returns
        -------
        org: str
        """
        org = self.org
        if org is None:
            org = _Config.get_openai_org()
        if org is None:
            raise RuntimeError("OpenAI organization was not found")
        return org


class GPTTextCompletionMixin(GPTMixin, BaseTextCompletionMixin):
    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """Gets a chat completion from the OpenAI API.

        Parameters
        ----------
        model : str
            The model to use.
        messages : Union[str, List[Dict[str, str]]]
            input messages to use.
        system_message : Optional[str]
            A system message to use.
        **kwargs : Any
            placeholder.

        Returns
        -------
        completion : dict
        """
        msgs = []
        if system_message is not None:
            msgs.append(construct_message("system", system_message))
        if isinstance(messages, str):
            msgs.append(construct_message("user", messages))
        else:
            for message in messages:
                msgs.append(construct_message(message["role"], message["content"]))
        completion = get_chat_completion(
            msgs,
            self._get_openai_key(),
            self._get_openai_org(),
            model,
            json_response=self._prefer_json_output,
        )
        return completion

    def _convert_completion_to_str(self, completion: Mapping[str, Any]):
        if hasattr(completion, "__getitem__"):
            return str(completion["choices"][0]["message"]["content"])
        return str(completion.choices[0].message.content)


class GPTClassifierMixin(GPTTextCompletionMixin, BaseClassifierMixin):
    _prefer_json_output = True

    def _extract_out_label(self, completion: Mapping[str, Any], **kwargs) -> Any:
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
            if hasattr(completion, "__getitem__"):
                label = extract_json_key(
                    completion["choices"][0]["message"]["content"], "label"
                )
            else:
                label = extract_json_key(completion.choices[0].message.content, "label")
        except Exception as e:
            print(completion)
            print(f"Could not extract the label from the completion: {str(e)}")
            label = ""
        return label


class GPTEmbeddingMixin(GPTMixin, BaseEmbeddingMixin):
    def _get_embeddings(self, text: np.ndarray) -> List[List[float]]:
        """Gets embeddings from the OpenAI compatible API.

        Parameters
        ----------
        text : str
            The text to embed.
        model : str
            The model to use.
        batch_size : int, optional
            The batch size to use. Defaults to 1.

        Returns
        -------
        embedding : List[List[float]]
        """
        embeddings = []
        print("Batch size:", self.batch_size)
        for i in tqdm(range(0, len(text), self.batch_size)):
            batch = text[i : i + self.batch_size].tolist()
            embeddings.extend(
                get_embedding(
                    batch,
                    self._get_openai_key(),
                    self._get_openai_org(),
                    self.model,
                )
            )

        return embeddings


# for now this works only with OpenAI
class GPTTunableMixin(BaseTunableMixin):
    _supported_tunable_models = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo"]

    def _build_label(self, label: str):
        return json.dumps({"label": label})

    def _set_hyperparameters(self, base_model: str, n_epochs: int, custom_suffix: str):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.custom_suffix = custom_suffix
        if base_model not in self._supported_tunable_models:
            raise ValueError(
                f"Model {base_model} is not supported. Supported models are"
                f" {self._supported_tunable_models}"
            )

    def _tune(self, X, y):
        if self.base_model.startswith(("azure::", "gpt4all")):
            raise ValueError(
                "Tuning is not supported for this model. Please use a different model."
            )
        file_uuid = str(uuid.uuid4())
        filename = f"skllm_{file_uuid}.jsonl"
        with open(filename, "w+") as f:
            for xi, yi in zip(X, y):
                prompt = self._get_prompt(xi)
                if not isinstance(prompt["messages"], str):
                    raise ValueError(
                        "Incompatible prompt. Use a prompt with a single message."
                    )
                f.write(
                    _build_clf_example(
                        prompt["messages"],
                        self._build_label(yi),
                        prompt["system_message"],
                    )
                )
                f.write("\n")
        client = _set_credentials_openai(self._get_openai_key(), self._get_openai_org())
        job = create_tuning_job(
            client,
            self.base_model,
            filename,
            self.n_epochs,
            self.custom_suffix,
        )
        print(f"Created new tuning job. JOB_ID = {job.id}")
        job = await_results(client, job.id)
        self.openai_model = job.fine_tuned_model
        self.model = self.openai_model  # openai_model is probably not required anymore
        delete_file(client, job.training_file)
        print(f"Finished training.")