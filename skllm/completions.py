from skllm.gpt4all_client import get_chat_completion as _g4a_get_chat_completion
from skllm.openai.chatgpt import get_chat_completion as _oai_get_chat_completion


def get_chat_completion(
    messages, openai_key=None, openai_org=None, model="gpt-3.5-turbo", max_retries=3
):
    if model.startswith("gpt4all::"):
        return _g4a_get_chat_completion(messages, model[9:])
    else:
        return _oai_get_chat_completion(
            messages, openai_key, openai_org, model, max_retries
        )
