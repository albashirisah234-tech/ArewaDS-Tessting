"""Tests for learning repository examples."""

import unittest

from learning_repository.machine_learning import SimpleLinearRegression
from learning_repository.python_basics import classify_number, is_even


class TestPythonBasics(unittest.TestCase):
    def test_classify_number(self) -> None:
        self.assertEqual(classify_number(10), "positive")
        self.assertEqual(classify_number(-2), "negative")
        self.assertEqual(classify_number(0), "zero")

    def test_is_even(self) -> None:
        self.assertTrue(is_even(8))
        self.assertFalse(is_even(7))


class TestMachineLearning(unittest.TestCase):
    def test_simple_linear_regression_fit_and_predict(self) -> None:
        model = SimpleLinearRegression()
        model.fit([1, 2, 3, 4], [2, 4, 6, 8])

        self.assertAlmostEqual(model.slope, 2.0)
        self.assertAlmostEqual(model.intercept, 0.0)
        self.assertAlmostEqual(model.predict(5), 10.0)

    def test_simple_linear_regression_validates_input(self) -> None:
        model = SimpleLinearRegression()

        with self.assertRaises(ValueError):
            model.fit([1], [2])

        with self.assertRaises(ValueError):
            model.fit([2, 2, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

