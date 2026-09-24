import unittest
import pandas as pd
import numpy as np
import importlib
import math
from src.ml_pipeline import MachineLearningPipeline
from src.utils import load_pipeline_config
# from statsmodels.regression.linear_model import RegressionResultsWrapper
# from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class TestMachineLearningPipeline(unittest.TestCase):
    """
    Class to test the ModelPipeline
    """

    # def setUp(self):
    #     """
    #     Import input data within the unit test pre-redfined setUp
    #     """
    #     self.data = pd.DataFrame(load_diabetes()["data"])
    #     self.data.columns = load_diabetes()["feature_names"]
    #     self.target = pd.DataFrame(load_diabetes()["target"])
    #     self.target.columns = ["target"]

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
        self.config_dict = load_pipeline_config("test_config.yaml")
        self.tune_one_model = self.config_dict["experiments"][0]["class"]

        # Dynamically import the module (e.g., 'sklearn.ensemble') for testing grid search
        imported_module = importlib.import_module(self.config_dict["experiments"][0]["module"])
        # This follwoing line acts like 'from sklearn.ensemble import RandomForestRegressor'
        model_class = getattr(imported_module, self.config_dict["experiments"][0]["class"])

        self.pipeline = MachineLearningPipeline(config=self.config_dict, model_estimator=self.valid_model)
        self.one_model_pipeline = MachineLearningPipeline(config=self.config_dict, model_estimator=model_class())

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
            MachineLearningPipeline(config=self.config_dict, model_estimator=LinearRegression)

        # Passing an invalid data type string
        with self.assertRaises(TypeError):
            MachineLearningPipeline(config=self.config_dict, model_estimator="NotAModel")

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

    def test_tune_single_model(self):
            """
            Verify the method executes a real grid search and returns a valid, fitted result.
            """

            # Access the target param_grid from the first experiment
            param_grid = self.config_dict["experiments"][0]["param_grid"]

            # Run the optimisation pipeline with cv=2 for speed
            grid_search_result = self.one_model_pipeline.tune_single_model(
                param_grid=param_grid,
                X_train=self.X_train,
                y_train=self.y_train,
                cv=2,
                scoring='neg_mean_squared_error'
            )

            # Assert 1: The return value is actually a fitted GridSearchCV object
            self.assertIsNotNone(grid_search_result)
            self.assertTrue(hasattr(grid_search_result, "best_estimator_"))
            self.assertTrue(hasattr(grid_search_result, "cv_results_"))

            # Assert 2: Verify parameter mapping worked under the hood
            self.assertIn('model__n_estimators', grid_search_result.cv_results_['params'][0])
            self.assertIn('model__max_depth', grid_search_result.cv_results_['params'][0])

            # Assert 3: Ensure the underlying pipeline can make real predictions
            best_pipeline = grid_search_result.best_estimator_
            predictions = best_pipeline.predict(self.X_train)
            self.assertEqual(len(predictions), len(self.y_train))

    def test_tune_single_model_empty_params(self):
        """Verify the function handles an empty param_grid without crashing."""
        blank_params = {}

        # Act: Run the method with an empty parameter grid
        result = self.one_model_pipeline.tune_single_model(
            param_grid=blank_params,
            X_train=self.X_train,
            y_train=self.y_train,
            cv=2,
            scoring='neg_mean_squared_error'
        )

        # Assert: It should still return a valid, fitted GridSearchCV object
        self.assertIsInstance(result, GridSearchCV)
        
        # Verify the reformatted param_grid is empty
        self.assertEqual(result.param_grid, {})

        # Verify it still executed a single cross-validation run on default settings
        self.assertTrue(hasattr(result, 'best_estimator_'))
        self.assertTrue(hasattr(result, 'best_params_'))
        self.assertEqual(result.best_params_, {})  # No custom params to report

        # Verify the grid search completed successfully (it is a valid number, not NaN)
        # and score is populated (not None)
        self.assertTrue(math.isfinite(result.best_score_), "GridSearchCV returned NaN or infinite score.")
        self.assertIsNotNone(result.best_score_)

    def test_tune_single_model_blank_estimator_raises_error(self):
        """Verify that running the tuner without a valid model estimator raises an error."""
        # 1. Arrange: Explicitly set the internal model estimator to None
        self.one_model_pipeline.model = None
        param_grid = self.config_dict["experiments"][0]["param_grid"]

        # 2. Act & Assert: Verify that executing the function raises a TypeError or ValueError
        # (Using a tuple allows the test to pass if scikit-learn raises either type)
        with self.assertRaises((TypeError, ValueError)) as context:
            self.one_model_pipeline.tune_single_model(
                param_grid=param_grid,
                X_train=self.X_train,
                y_train=self.y_train,
                cv=2,
                scoring='accuracy'
            )
        self.assertIn("estimator", str(context.exception).lower())

if __name__ == "__main__":
    unittest.main()
