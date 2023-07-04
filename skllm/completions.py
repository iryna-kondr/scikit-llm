from skllm.gpt4all_client import get_chat_completion as _g4a_get_chat_completion
from skllm.openai.chatgpt import get_chat_completion as _oai_get_chat_completion


def get_chat_completion(
    messages: dict,
    openai_key: str = None,
    openai_org: str = None,
    model: str = "gpt-3.5-turbo",
    max_retries: int = 3,
):
    """Gets a chat completion from the OpenAI API."""
    if model.startswith("gpt4all::"):
        return _g4a_get_chat_completion(messages, model[9:])
    else:
        api = "azure" if model.startswith("azure::") else "openai"
        if api == "azure":
            model = model[7:]
        return _oai_get_chat_completion(
            messages, openai_key, openai_org, model, max_retries, api=api
        )
