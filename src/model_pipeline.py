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

    def linearity_plots(self):
        """
        """
        # Check that initial_fitted_values and residual exist (initial_model) has been run

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Plot A: Linearity & Homoscedasticity (Residuals vs Fitted)
        sns.scatterplot(x=self.initial_fitted_values, y=self.initial_residuals, ax=axes[0, 0], color='purple', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals vs Fitted (Linearity & Homoscedasticity Check)')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        
        # Plot B: Normality (Q-Q Plot)
        qqplot(self.initial_residuals, line='45', fit=True, ax=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot (Normality Check)')
        
        # Plot C: Normality Distribution (Histogram)
        sns.histplot(self.initial_residuals, kde=True, ax=axes[1, 0], color='teal')
        axes[1, 0].set_title('Histogram of Residuals')
        axes[1, 0].set_xlabel('Residual Error')
        
        # Plot D: Independence (Residual Order Plot)
        axes[1, 1].plot(self.initial_residuals.index, self.initial_residuals, marker='o', linestyle='', color='orange', alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals vs Order (Independence Check)')
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel('Residuals')

        self.initial_fig = fig