import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS

class VIFAndPValueSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=5.0, p_threshold=0.05):
        self.vif_threshold = vif_threshold
        self.p_threshold = p_threshold
        self.final_features_ = None
    
    def fit(self, X, y):
        # Check of dataframe type? 
        # TO DO
        
        # Convert to DataFrame to safely keep track of column indices
        X_df = pd.DataFrame(X).copy()
        
        # --- STEP 1: Iterative VIF Elimination ---
        while True:
            # VIF requires an intercept column to be calculated properly
            X_with_const = add_constant(X_df, has_constant='add')
            
            # Calculate VIF for each feature (skip the constant column index)
            vif_data = []
            for i in range(X_df.shape[1]):
                col_idx = X_with_const.columns.get_loc(X_df.columns[i])
                vif = variance_inflation_factor(X_with_const.values, col_idx)
                vif_data.append(vif)
                
            vif_series = pd.Series(vif_data, index=X_df.columns)
            max_vif = vif_series.max()
            
            # If the highest VIF exceeds threshold, drop that column and recalculate
            if max_vif > self.vif_threshold and X_df.shape[1] > 1:
                max_vif_column = vif_series.idxmax()
                X_df = X_df.drop(columns=[max_vif_column])
            else:
                break
                
        # --- STEP 2: P-Value Selection on Remaining Features ---
        X_final_with_const = add_constant(X_df, has_constant='add')
        self.ols_model = OLS(y, X_final_with_const).fit()
        
        # Extract p-values (dropping the constant row)
        p_values = self.ols_model.pvalues.drop('const', errors='ignore')
        
        # Filter features that survive the significance threshold
        self.final_features_ = p_values[p_values <= self.p_threshold].index.tolist()
        
        # Edge case: If nothing is significant, keep at least the single best feature
        if not self.final_features_ and not X_df.empty:
            self.final_features_ = [p_values.idxmin()]
            
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        # Filter down to the exact feature subset decided during fit()
        return X_df[self.final_features_]#.to_numpy()