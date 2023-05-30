try:
    from gpt4all import GPT4All
except (ImportError, ModuleNotFoundError):
    GPT4All = None

_loaded_models = {}


def get_chat_completion(messages, model="ggml-gpt4all-j-v1.3-groovy"):
    if GPT4All is None:
        raise ImportError(
            "gpt4all is not installed, try `pip install scikit-llm[gpt4all]`"
        )
    if model not in _loaded_models.keys():
        _loaded_models[model] = GPT4All(model)
    return _loaded_models[model].chat_completion(
        messages, verbose=False, streaming=False, temp=1e-10
    )


def unload_models():
    global _loaded_models
    _loaded_models = {}
