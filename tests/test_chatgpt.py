import unittest
from unittest.mock import patch

from skllm.openai.chatgpt import (
    construct_message,
    extract_json_key,
    get_chat_completion,
)


class TestChatGPT(unittest.TestCase):

    @patch("skllm.openai.credentials.set_credentials")
    @patch("openai.ChatCompletion.create")
    def test_get_chat_completion(self, mock_create, mock_set_credentials):
        messages = [{"role": "system", "content": "Hello"}]
        key = "some_key"
        org = "some_org"
        model = "gpt-3.5-turbo"
        mock_create.side_effect = [Exception("API error"), "success"]

        result = get_chat_completion(messages, key, org, model)

        self.assertTrue(mock_set_credentials.call_count <= 1, "set_credentials should be called at most once")
        self.assertEqual(mock_create.call_count, 2, "ChatCompletion.create should be called twice due to an exception "
                                                    "on the first call")
        self.assertEqual(result, "success")

    def test_construct_message(self):
        role = "user"
        content = "Hello, World!"
        message = construct_message(role, content)
        self.assertEqual(message, {"role": role, "content": content})
        with self.assertRaises(ValueError):
            construct_message("invalid_role", content)

    def test_extract_json_key(self):
        json_ = '{"key": "value"}'
        key = "key"
        result = extract_json_key(json_, key)
        self.assertEqual(result, "value")

        # Given that the function returns None when a KeyError occurs, adjust the assertion
        result_with_invalid_key = extract_json_key(json_, "invalid_key")
        self.assertEqual(result_with_invalid_key, None)


if __name__ == '__main__':
    unittest.main()
