import os
from typing import Optional

_OPENAI_KEY_VAR = 'SLLM_CONFIG_OPENAI_KEY'

class SLLMConfig():
    
    @staticmethod
    def set_openai_key(key: str) -> None:
        os.environ[_OPENAI_KEY_VAR] = key

    @staticmethod
    def get_openai_key() -> Optional[str]:
        return os.environ.get(_OPENAI_KEY_VAR, None)