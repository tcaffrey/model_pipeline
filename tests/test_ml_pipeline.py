import unittest
import pandas as pd
import numpy as np
from src.ml_pipeline import MachineLearningPipeline
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


class TestMachineLearningPipeline(unittest.TestCase):
    """
    Class to test the ModelPipeline
    """

    def setUp(self):
        """
        Import input data within the unit test pre-redfined setUp
        """
        self.data = pd.DataFrame(load_diabetes()["data"])
        self.data.columns = load_diabetes()["feature_names"]
        self.target = pd.DataFrame(load_diabetes()["target"])
        self.target.columns = ["target"]

    def setUp(self):
        """Runs before every individual test. Sets up fresh mock data and pipelines."""
        # Clean training data
        self.X_train = pd.DataFrame(
            {
                "age": [25.0, 47.0, 38.0, 22.0],
                "city": ["London", "Paris", "London", "Berlin"],
            }
        )
        self.y_train = np.array([1, 0, 1, 0])

        # Test data with unseen and missing values
        self.X_test = pd.DataFrame(
            {"age": [30.0, np.nan], "city": [" PARIS ", "Tokyo"]}
        )

        # Standard valid model
        self.valid_model = LinearRegression()
        self.pipeline = MachineLearningPipeline(self.valid_model)

    def test_initialisation_with_valid_model(self):
        """
        Verify the pipeline initialises cleanly with a valid estimator.
        """
        self.assertEqual(self.pipeline.model, self.valid_model)
        self.assertFalse(self.pipeline.is_fitted)
        self.assertIsNone(self.pipeline.preprocessor)

    def test_initialisation_with_invalid_model_raises_error(self):
        """
        Verify that passing an uninstantiated class or raw string triggers a TypeError.
        """
        # Passing class instead of instance
        with self.assertRaises(TypeError):
            MachineLearningPipeline(model_estimator=LinearRegression)

        # Passing an invalid data type string
        with self.assertRaises(TypeError):
            MachineLearningPipeline(model_estimator="NotAModel")

    def test_static_validation_method_directly(self):
        """
        Verify the @staticmethod validation works independently without an instance.
        """
        # Test valid input passes cleanly
        result = MachineLearningPipeline._validate_model(self.valid_model)
        self.assertEqual(result, self.valid_model)

        # Test invalid input raises exception
        with self.assertRaises(TypeError):
            MachineLearningPipeline._validate_model([])

    def test_predict_before_fit_raises_error(self):
        """
        Verify calling predict before fitting throws an explicit error.
        """
        with self.assertRaises(ValueError) as context:
            self.pipeline.predict(self.X_test)

        self.assertIn("Pipeline must be fitted", str(context.exception))

    def test_successful_fit_and_predict_flow(self):
        """
        Verify the end-to-end flow executes and alters internal flags accurately.
        """
        returned_pipeline = self.pipeline.fit(self.X_train, self.y_train)

        self.assertTrue(self.pipeline.is_fitted)
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertEqual(returned_pipeline, self.pipeline)

        predictions = self.pipeline.predict(self.X_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))


if __name__ == "__main__":
    unittest.main()
