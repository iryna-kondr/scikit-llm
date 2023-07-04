import os
from typing import Optional

_OPENAI_KEY_VAR = "SKLLM_CONFIG_OPENAI_KEY"
_OPENAI_ORG_VAR = "SKLLM_CONFIG_OPENAI_ORG"
_AZURE_API_BASE_VAR = "SKLLM_CONFIG_AZURE_API_BASE"
_AZURE_API_VERSION_VAR = "SKLLM_CONFIG_AZURE_API_VERSION"
_GOOGLE_PROJECT = "GOOGLE_CLOUD_PROJECT"


class SKLLMConfig:
    @staticmethod
    def set_openai_key(key: str) -> None:
        """Sets the OpenAI key.

        Parameters
        ----------
        key : str
            OpenAI key.
        """
        os.environ[_OPENAI_KEY_VAR] = key

    @staticmethod
    def get_openai_key() -> Optional[str]:
        """Gets the OpenAI key.

        Returns
        -------
        Optional[str]
            OpenAI key.
        """
        return os.environ.get(_OPENAI_KEY_VAR, None)

    @staticmethod
    def set_openai_org(key: str) -> None:
        """Sets OpenAI organization ID.

        Parameters
        ----------
        key : str
            OpenAI organization ID.
        """
        os.environ[_OPENAI_ORG_VAR] = key

    @staticmethod
    def get_openai_org() -> str:
        """Gets the OpenAI organization ID.

        Returns
        -------
        str
            OpenAI organization ID.
        """
        return os.environ.get(_OPENAI_ORG_VAR, "")

    @staticmethod
    def get_azure_api_base() -> str:
        """Gets the API base for Azure.

        Returns
        -------
        str
            URL to be used as the base for the Azure API.
        """
        base = os.environ.get(_AZURE_API_BASE_VAR, None)
        if base is None:
            raise RuntimeError("Azure API base is not set")
        return base

    @staticmethod
    def set_azure_api_base(base: str) -> None:
        """Set the API base for Azure.

        Parameters
        ----------
        base : str
            URL to be used as the base for the Azure API.
        """
        os.environ[_AZURE_API_BASE_VAR] = base

    @staticmethod
    def set_azure_api_version(ver: str) -> None:
        """Set the API version for Azure.

        Parameters
        ----------
        ver : str
            Azure API version.
        """
        os.environ[_AZURE_API_VERSION_VAR] = ver

    @staticmethod
    def get_azure_api_version() -> str:
        """Gets the API version for Azure.

        Returns
        -------
        str
            Azure API version.
        """
        return os.environ.get(_AZURE_API_VERSION_VAR, "2023-05-15")

    @staticmethod
    def get_google_project() -> Optional[str]:
        """Gets the Google Cloud project ID.

        Returns
        -------
        Optional[str]
            Google Cloud project ID.
        """
        return os.environ.get(_GOOGLE_PROJECT, None)

    @staticmethod
    def set_google_project(project: str) -> None:
        """Sets the Google Cloud project ID.

        Parameters
        ----------
        project : str
            Google Cloud project ID.
        """
        os.environ[_GOOGLE_PROJECT] = project
