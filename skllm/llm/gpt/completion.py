import warnings
from skllm.llm.gpt.clients.openai.completion import (
    get_chat_completion as _oai_get_chat_completion,
)
from skllm.llm.gpt.clients.gpt4all.completion import (
    get_chat_completion as _g4a_get_chat_completion,
)
from skllm.llm.gpt.utils import split_to_api_and_model
from skllm.config import SKLLMConfig as _Config

def get_chat_completion(
    messages: dict,
    openai_key: str = None,
    openai_org: str = None,
    model: str = "gpt-3.5-turbo",
    json_response: bool = False,
):
    """Gets a chat completion from the OpenAI compatible API."""
    api, model = split_to_api_and_model(model)
    if api == "gpt4all":
        return _g4a_get_chat_completion(messages, model)
    else:
        url = _Config.get_gpt_url()
        if api == "openai" and url is not None:
            warnings.warn(f"You are using the OpenAI backend with a custom URL: {url}; did you mean to use the `custom_url` backend?\nTo use the OpenAI backend, please remove the custom URL using `SKLLMConfig.reset_gpt_url()`.")
        elif api == "custom_url" and url is None:
            raise ValueError("You are using the `custom_url` backend but no custom URL was provided. Please set it using `SKLLMConfig.set_gpt_url(<url>)`.")
        return _oai_get_chat_completion(
            messages,
            openai_key,
            openai_org,
            model,
            api=api,
            json_response=json_response,
        )
