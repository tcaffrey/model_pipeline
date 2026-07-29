import unittest
import pandas as pd
from src.model_pipeline import ModelPipeline
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.datasets import load_diabetes


class TestModelPipeline(unittest.TestCase):
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

    def tearDown(self):
        print("Running tearDown method...")

    def test_initial_model(self):
        """
        Test to determine if the initial model for linear regression is
        fitted successfully and can be called.
        """
        model_pipeline = ModelPipeline(self.data, self.target)
        model_pipeline.initial_model()
        self.assertIsInstance(model_pipeline.initial_model, RegressionResultsWrapper)


if __name__ == "__main__":
    unittest.main()
