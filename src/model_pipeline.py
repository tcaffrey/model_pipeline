from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS


class ModelPipeline:

    def __init__(self, data, target):
        self.data = data
        self.target = target

    # check data and target are in correct form
    def initial_model(self):

        X = self.data.copy()
        y = self.target.copy()
        X = add_constant(X)

        self.initial_model = OLS(y, X).fit()
        self.initial_residuals = self.initial_model.resid
        self.initial_fitted_values = self.initial_model.fittedvalues
