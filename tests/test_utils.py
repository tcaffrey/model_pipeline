import unittest
from unittest.mock import patch, MagicMock
from src.utils import (
    detect_numerical_categorical_features,
    linearity_plots,
    adaptive_categorical_transformer,
    initial_model,
)
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder


class TestUtils(unittest.TestCase):
    def setUp(self):
        """
        Import input data within the unit test pre-redfined setUp
        """
        self.data = pd.DataFrame(load_diabetes()["data"])
        self.data.columns = load_diabetes()["feature_names"]
        self.target = pd.DataFrame(load_diabetes()["target"])
        self.target.columns = ["target"]

        # Set up mock data with distinct cardinality tiers
        low_card_data = ["Low", "Medium", "High"] * 10
        high_card_data = [f"ID_{i}" for i in range(15)] * 2
        self.target_enc_X = pd.DataFrame(
            {
                "Priority": low_card_data,
                "IDCode": high_card_data,
                "IgnoredCol": ["Ignore"] * 30,
            }
        )
        # Adding a missing value to test how functionality handles this.
        self.target_enc_X.loc[0, "Priority"] = np.nan 
        self.categorical_features = ["Priority", "IDCode"]
        self.regression_flag = True
        self.X_ols = pd.Series([1, 2, 3, 4, 5], name="Feature")
        self.y_ols = pd.Series([7.1, 8.9, 11.2, 12.8, 15.0], name="Target")

    def test_detect_numerical_categorical_features(self):
        """
        Verify that the function correctly returns whether features in a 
        dataset are numerical or categorical given different types of inputs.        
        """
        input_1 = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": ["one", "two", "three"],
                "feature_3": [True, False, True],
                "feature_4": [None, None, None],
            }
        )
        input_2 = pd.DataFrame({})
        input_3 = ["123", 123]
        input_4 = 123

        expected_1_num, expected_1_cat = ["feature_1", "feature_3"], [
            "feature_2",
            "feature_4",
        ]
        expected_2_num, expected_2_cat = [], []
        expected_3_num, expected_3_cat = None, None
        expected_4_num, expected_4_cat = None, None

        actual_1_num, actual_1_cat = detect_numerical_categorical_features(input_1)
        actual_2_num, actual_2_cat = detect_numerical_categorical_features(input_2)
        actual_3_num, actual_3_cat = detect_numerical_categorical_features(input_3)
        actual_4_num, actual_4_cat = detect_numerical_categorical_features(input_4)

        self.assertEqual([actual_1_num, actual_1_cat], [expected_1_num, expected_1_cat])
        self.assertEqual([actual_2_num, actual_2_cat], [expected_2_num, expected_2_cat])
        self.assertEqual([actual_3_num, actual_3_cat], [expected_3_num, expected_3_cat])
        self.assertEqual([actual_4_num, actual_4_cat], [expected_4_num, expected_4_cat])

    @patch('src.utils.OLS')
    @patch('src.utils.add_constant')
    def test_initial_model_execution_flow(self, mock_add_constant, mock_ols):
        """Verify that statsmodels functions are called with correctly."""

        mock_X_with_constant = MagicMock()
        mock_add_constant.return_value = mock_X_with_constant

        mock_ols_instance = MagicMock()
        mock_ols.return_value = mock_ols_instance

        # Call the function
        initial_model(self.X_ols, self.y_ols)

        # Verify add_constant was called with X
        mock_add_constant.assert_called_once_with(self.X_ols)

        # Verify OLS was instantiated with y and the modified X
        mock_ols.assert_called_once_with(self.y_ols, mock_X_with_constant)

        # Verify fit() was called on the OLS instance
        mock_ols_instance.fit.assert_called_once()

    def test_initial_model_values(self):
        """Verify that the initial_model function will return expected values."""

        # Mock the properties the initial_model function extracts
        expected_fitted = pd.Series([7.06, 9.03, 11.00, 12.97, 14.94])
        expected_residuals = pd.Series([0.04, -0.13, 0.20, -0.17, 0.06])

        # Verify the returned tuple contents match what the model provided
        fitted_model = initial_model(self.X_ols, self.y_ols)
        pd.testing.assert_series_equal(fitted_model.fittedvalues, expected_fitted)
        pd.testing.assert_series_equal(fitted_model.resid, expected_residuals)

    def test_linearity_plots(self):
        """
        Verify that plots are created for testing linearity of data. Sample data
        is fitted to an initial linear model to generated fitted values and residuals.
        """
        fitted_model = initial_model(self.data, self.target)
        fig, ax = linearity_plots(
            fitted_model.fittedvalues, fitted_model.resid
        )

        self.assertIsInstance(fig, mfigure.Figure)
        self.assertIsInstance(ax[0, 0], maxes.Axes)

    def test_target_enc_pipeline_structure_and_output_type(self):
        """
        Verify the adapative categorical function returns a Pipeline configured
        for pandas outputs.
        """
        pipeline = adaptive_categorical_transformer(
            df=self.target_enc_X,
            regression_flag=self.regression_flag,
            categorical_features=self.categorical_features,
        )

        # Assert structural wrapper types
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(pipeline.named_steps["encoder"], ColumnTransformer)

        # Verify the critical set_output tracking configuration
        y_mock = np.random.randn(len(self.target_enc_X))
        processed_df = pipeline.fit_transform(self.target_enc_X, y_mock)
        self.assertIsInstance(processed_df, pd.DataFrame)

    def test_target_enc_low_and_high_cardinality(self):
        """
        Confirm features are correctly split between OneHotEncoder and TargetEncoder.
        """
        pipeline = adaptive_categorical_transformer(
            df=self.target_enc_X,
            regression_flag=self.regression_flag,
            categorical_features=self.categorical_features,
            cardinality_threshold=10,
        )

        col_transformer = pipeline.named_steps["encoder"]
        transformers_dict = {
            name: (trans, cols) for name, trans, cols in col_transformer.transformers
        }

        # Verify columns are passed to the correct processing block
        self.assertIn("one_hot", transformers_dict)
        self.assertIn("target_enc", transformers_dict)
        self.assertEqual(transformers_dict["one_hot"][1], ["Priority"])
        self.assertEqual(transformers_dict["target_enc"][1], ["IDCode"])

        # Check specific encoder configurations
        self.assertIsInstance(transformers_dict["one_hot"][0], OneHotEncoder)
        self.assertIsInstance(transformers_dict["target_enc"][0], TargetEncoder)
        self.assertEqual(transformers_dict["one_hot"][0].drop, "if_binary")
        self.assertEqual(transformers_dict["one_hot"][0].handle_unknown, "ignore")
        self.assertEqual(transformers_dict["target_enc"][0].target_type, "continuous")
        self.assertEqual(transformers_dict["target_enc"][0].cv, 5)

    def test_target_enc_low_cardinality_features_present_only(self):
        """
        Verify the transformer handles scenarios where no features exceed the threshold.
        """
        # Force threshold to be higher than any column cardinality
        pipeline = adaptive_categorical_transformer(
            df=self.target_enc_X,
            regression_flag=self.regression_flag,
            categorical_features=self.categorical_features,
            cardinality_threshold=100,
        )

        col_transformer = pipeline.named_steps["encoder"]
        active_transformer_names = [
            name for name, _, _ in col_transformer.transformers if name != "remainder"
        ]

        self.assertIn("one_hot", active_transformer_names)
        self.assertNotIn("target_enc", active_transformer_names)

    def test_target_enc_high_cardinality_features_present_only(self):
        """
        Verify the transformer handles scenarios where all features exceed the threshold.
        """
        # Force threshold to be lower than any column cardinality
        pipeline = adaptive_categorical_transformer(
            df=self.target_enc_X,
            regression_flag=self.regression_flag,
            categorical_features=self.categorical_features,
            cardinality_threshold=1,
        )

        col_transformer = pipeline.named_steps["encoder"]
        active_transformer_names = [
            name for name, _, _ in col_transformer.transformers if name != "remainder"
        ]

        self.assertNotIn("one_hot", active_transformer_names)
        self.assertIn("target_enc", active_transformer_names)

    def test_target_enc_execution_stability(self):
        """
        Verify that fit_transform runs without type errors or NaN leakages.
        """
        pipeline = adaptive_categorical_transformer(
            df=self.target_enc_X,
            regression_flag=self.regression_flag,
            categorical_features=self.categorical_features,
            cardinality_threshold=10,
            cv=None,
        )

        # Synthetic target created, matching the length of the sample data
        y_mock = np.random.randn(len(self.target_enc_X))

        try:
            processed_df = pipeline.fit_transform(self.target_enc_X, y_mock)
        except Exception as e:
            self.fail(
                f"The generated pipeline crashed during standard execution! Error: {e}"
            )

        # Assert type output
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertFalse(
            processed_df.isna().any().any(), "Imputer failed to resolve missing values."
        )
        # Ensure non-categorical columns were correctly dropped via the 'remainder' parameter
        self.assertNotIn("IgnoredCol", processed_df.columns)


if __name__ == "__main__":
    unittest.main()
