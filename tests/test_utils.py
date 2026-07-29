import unittest
from src.model_pipeline import ModelPipeline
from src.utils import (
    detect_numerical_categorical_features,
    linearity_plots,
)
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt

class TestUtils(unittest.TestCase):
    def setUp(self):
        """
        Import input data within the unit test pre-redfined setUp
        """
        self.data = pd.DataFrame(load_diabetes()["data"])
        self.data.columns = load_diabetes()["feature_names"]
        self.target = pd.DataFrame(load_diabetes()["target"])
        self.target.columns = ["target"]
        # test_data_path = "/".join(["tests", "test_data"])

        # self.data = pd.read_csv(f"{test_data_path}/data.csv")

    def tearDown(self):
        print("Running tearDown method...")

    def test_detect_numerical_categorical_features(self):
        """
        Verify that the function to calculate whether the features in a dataset are 
        numerical or categorical given different types of inputs.
        """
        input_1 = pd.DataFrame({"feature_1": [1,2,3],
                               "feature_2": ["one", "two", "three"],
                               "feature_3": [True, False, True],
                               "feature_4": [None, None, None]})   
        input_2 = pd.DataFrame({})  
        input_3 = ["123", 123]
        input_4 = 123
    
        expected_1_num, expected_1_cat = ["feature_1", "feature_3"], ["feature_2", "feature_4"]
        expected_2_num, expected_2_cat = [], []
        expected_3_num, expected_3_cat = None, None
        expected_4_num, expected_4_cat = None, None
    
        actual_1_num, actual_1_cat = detect_numerical_categorical_features(input_1)
        actual_2_num, actual_2_cat = detect_numerical_categorical_features(input_2)
        actual_3_num, actual_3_cat = detect_numerical_categorical_features(input_3)
        actual_4_num, actual_4_cat = detect_numerical_categorical_features(input_4)
        
        self.assertEqual([actual_1_num, actual_1_cat],[expected_1_num, expected_1_cat])
        self.assertEqual([actual_2_num, actual_2_cat],[expected_2_num, expected_2_cat])
        self.assertEqual([actual_3_num, actual_3_cat],[expected_3_num, expected_3_cat])
        self.assertEqual([actual_4_num, actual_4_cat],[expected_4_num, expected_4_cat])

    def test_linearity_plots(self):
        """
        Verify that plots are created for testing linearity of data. Sample data
        is fitted to an initial linear model to generated fitted values and residuals.
        """
        model_pipeline = ModelPipeline(self.data , self.target)
        model_pipeline.initial_model()
        fig, ax = linearity_plots(model_pipeline.initial_fitted_values, model_pipeline.initial_residuals)

        self.assertIsInstance(fig, mfigure.Figure)
        self.assertIsInstance(ax[0, 0], maxes.Axes)
        

if __name__ == '__main__':
    unittest.main()