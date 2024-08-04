import threading
import os
import hashlib
import requests
from tqdm import tqdm
import hashlib
from typing import Optional
import tempfile
from skllm.config import SKLLMConfig
from warnings import warn


try:
    from llama_cpp import Llama as _Llama

    _llama_imported = True
except (ImportError, ModuleNotFoundError):
    _llama_imported = False


supported_models = {
    "llama3-8b-q4": {
        "download_url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "sha256": "c57380038ea85d8bec586ec2af9c91abc2f2b332d41d6cf180581d7bdffb93c1",
        "n_ctx": 8192,
        "supports_system_message": True,
    },
    "gemma2-9b-q4": {
        "download_url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "sha256": "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe",
        "n_ctx": 8192,
        "supports_system_message": False,
    },
    "phi3-mini-q4": {
        "download_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "sha256": "8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef",
        "n_ctx": 4096,
        "supports_system_message": False,
    },
    "mistral0.3-7b-q4": {
        "download_url": "https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "sha256": "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6",
        "n_ctx": 32768,
        "supports_system_message": False,
    },
    "gemma2-2b-q6": {
        "download_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q6_K_L.gguf",
        "sha256": "b2ef9f67b38c6e246e593cdb9739e34043d84549755a1057d402563a78ff2254",
        "n_ctx": 8192,
        "supports_system_message": False,
    },
}


class LlamaHandler:

    def maybe_download_model(self, model_name, download_url, sha256) -> str:
        download_folder = SKLLMConfig.get_gguf_download_path()
        os.makedirs(download_folder, exist_ok=True)
        model_name = model_name + ".gguf"
        model_path = os.path.join(download_folder, model_name)
        if not os.path.exists(model_path):
            print("The model `{0}` is not found locally.".format(model_name))
            self._download_model(model_name, download_folder, download_url, sha256)
        return model_path

    def _download_model(
        self, model_filename: str, model_path: str, url: str, expected_sha256: str
    ) -> str:
        full_path = os.path.join(model_path, model_filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=model_path)
        temp_path = temp_file.name
        temp_file.close()

        response = requests.get(url, stream=True)

        if response.status_code != 200:
            os.remove(temp_path)
            raise ValueError(
                f"Request failed: HTTP {response.status_code} {response.reason}"
            )

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024 * 4

        sha256 = hashlib.sha256()

        with (
            open(temp_path, "wb") as file,
            tqdm(
                desc="Downloading {0}: ".format(model_filename),
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
            ) as progress_bar,
        ):
            for data in response.iter_content(block_size):
                file.write(data)
                sha256.update(data)
                progress_bar.update(len(data))

        downloaded_sha256 = sha256.hexdigest()
        if downloaded_sha256 != expected_sha256:
            raise ValueError(
                f"Expected SHA-256 hash {expected_sha256}, but got {downloaded_sha256}"
            )

        os.rename(temp_path, full_path)

    def __init__(self, model: str):
        if not _llama_imported:
            raise ImportError(
                "llama_cpp is not installed, try `pip install scikit-llm[llama_cpp]`"
            )
        self.lock = threading.Lock()
        if model not in supported_models:
            raise ValueError(f"Model {model} is not supported.")
        download_url = supported_models[model]["download_url"]
        sha256 = supported_models[model]["sha256"]
        n_ctx = supported_models[model]["n_ctx"]
        self.supports_system_message = supported_models[model][
            "supports_system_message"
        ]
        if not self.supports_system_message:
            warn(
                f"The model {model} does not support system messages. This may cause issues with some estimators."
            )
        extended_model_name = model + "-" + sha256[:8]
        model_path = self.maybe_download_model(
            extended_model_name, download_url, sha256
        )
        max_gpu_layers = SKLLMConfig.get_gguf_max_gpu_layers()
        verbose = SKLLMConfig.get_gguf_verbose()
        self.model = _Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            verbose=verbose,
            n_gpu_layers=max_gpu_layers,
        )

    def get_chat_completion(self, messages: dict, **kwargs):
        if not self.supports_system_message:
            messages = [m for m in messages if m["role"] != "system"]
        with self.lock:
            return self.model.create_chat_completion(
                messages, temperature=0.0, **kwargs
            )


class ModelCache:
    lock = threading.Lock()
    cache: dict[str, LlamaHandler] = {}

    @classmethod
    def get(cls, key) -> Optional[LlamaHandler]:
        return cls.cache.get(key, None)

    @classmethod
    def store(cls, key, value):
        cls.cache[key] = value

    @classmethod
    def clear(cls):
        cls.cache = {}
