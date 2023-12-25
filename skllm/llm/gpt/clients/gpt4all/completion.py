from typing import Dict

try:
    from gpt4all import GPT4All
except (ImportError, ModuleNotFoundError):
    GPT4All = None

_loaded_models = {}


def _make_openai_compatabile(message: str) -> Dict:
    return {"choices": [{"message": {"content": message, "role": "assistant"}}]}


def get_chat_completion(
    messages: Dict, model: str = "ggml-model-gpt4all-falcon-q4_0.bin"
) -> Dict:
    """Gets a chat completion from GPT4All.

    Parameters
    ----------
    messages : Dict
        The messages to use as a prompt for the chat completion.
    model : str
        The model to use for the chat completion. Defaults to "ggml-gpt4all-j-v1.3-groovy".

    Returns
    -------
    completion : Dict
    """
    if GPT4All is None:
        raise ImportError(
            "gpt4all is not installed, try `pip install scikit-llm[gpt4all]`"
        )
    if model not in _loaded_models.keys():
        loaded_model = GPT4All(model)
        _loaded_models[model] = loaded_model
        loaded_model._current_prompt_template = loaded_model.config["promptTemplate"]

    prompt = _loaded_models[model]._format_chat_prompt_template(
        messages, _loaded_models[model].config["systemPrompt"]
    )
    generated = _loaded_models[model].generate(
        prompt,
        streaming=False,
        temp=1e-10,
    )

    return _make_openai_compatabile(generated)


def unload_models() -> None:
    global _loaded_models
    _loaded_models = {}
