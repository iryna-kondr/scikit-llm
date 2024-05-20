from typing import Tuple

SUPPORTED_APIS = ["openai", "azure", "gpt4all", "custom_url"]


def split_to_api_and_model(model: str) -> Tuple[str, str]:
    if "::" not in model:
        return "openai", model
    for api in SUPPORTED_APIS:
        if model.startswith(f"{api}::"):
            return api, model[len(api) + 2 :]
    raise ValueError(f"Unsupported API: {model.split('::')[0]}")