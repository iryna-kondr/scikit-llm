from typing import Optional
from skllm.config import SKLLMConfig as _Config

class OpenAIMixin:

    def _set_keys(self, key: Optional[str] = None, org: Optional[str] = None) -> None:
        self.openai_key = key
        self.openai_org = org

    def _get_openai_key(self) -> str:
        key = self.openai_key
        if key is None:
            key = _Config.get_openai_key()
        if key is None:
            raise RuntimeError("OpenAI key was not found")
        return key

    def _get_openai_org(self) -> str:
        key = self.openai_org
        if key is None:
            key = _Config.get_openai_org()
        if key is None:
            raise RuntimeError("OpenAI organization was not found")
        return key
 