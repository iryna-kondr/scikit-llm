import os
from typing import Optional

_OPENAI_KEY_VAR = "SKLLM_CONFIG_OPENAI_KEY"
_OPENAI_ORG_VAR = "SKLLM_CONFIG_OPENAI_ORG"

class SKLLMConfig():
    
    @staticmethod
    def set_openai_key(key: str) -> None:
        os.environ[_OPENAI_KEY_VAR] = key

    @staticmethod
    def get_openai_key() -> Optional[str]:
        return os.environ.get(_OPENAI_KEY_VAR, None)
    
    @staticmethod
    def set_openai_org(key: str) -> None:
        os.environ[_OPENAI_ORG_VAR] = key

    @staticmethod
    def get_openai_org() -> Optional[str]:
        return os.environ.get(_OPENAI_ORG_VAR, None)