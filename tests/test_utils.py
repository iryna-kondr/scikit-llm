import unittest
import pandas as pd
import numpy as np
from skllm import utils


class TestUtils(unittest.TestCase):
    def test_to_numpy(self):
        # Test with pandas Series
        series = pd.Series([1, 2, 3])
        result = utils.to_numpy(series)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.tolist(), [1, 2, 3])

        # Test with list
        list_data = [4, 5, 6]
        result = utils.to_numpy(list_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.tolist(), [4, 5, 6])

    def test_find_json_in_string(self):
        # Test with string containing JSON
        string = 'Hello {"name": "John", "age": 30} World'
        result = utils.find_json_in_string(string)
        self.assertEqual(result, '{"name": "John", "age": 30}')

        # Test with string without JSON
        string = "Hello World"
        result = utils.find_json_in_string(string)
        self.assertEqual(result, "{}")


if __name__ == "__main__":
    unittest.main()
