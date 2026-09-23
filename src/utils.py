import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.api import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor


def detect_numerical_categorical_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Function to detect the numerical and categorical features in a dataframe

    Args:
        df (pd.DataFrame): The dataframe to detect features types

    Returns:
        tuple: A tuple of two lists, one with numerical features,
        the other categorical features.
    """

    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        numerical_features = df.dtypes[
            (df.dtypes != "str") & (df.dtypes != object)
        ].index.tolist()
        categorical_features = df.dtypes[
            (df.dtypes == "str") | (df.dtypes == object)
        ].index.tolist()

        return numerical_features, categorical_features

    except TypeError as e:
        print(f"Type validation failed: {e}, returning None")
        return None, None


def initial_model(X: pd.Series, y: pd.Series):
    """
    For basic checks of linearity and to produce accompanying plots, an
    Ordinary Least Square model is fitted to the data and target values
    provided.

    Args:
        X (pd.Series): The feature set to use for training the model
        y (pd.Series): The target to train the model on

    Returns:
        tuple: A fitted OLS model.
    """

    X = add_constant(X)
    return OLS(y, X).fit()


def linearity_plots(fitted_values: pd.Series, residuals: pd.Series):
    """
    Function to produce a series of plots that help with detecting if data is
    displaying linear behaviour between features and target.

    Args:
        fitted_values (pd.Series): The values from the fitted linear model
        fitted_residuals (pd.Series): The residuals from the fitted linear model

    Returns:
        tuple: A tuple of the maplotlib figure and array of axes for the
        subplots.
    """
    # Check that fitted values and residuals are of 1-d array

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Plot A: Linearity & Homoscedasticity (Residuals vs Fitted)
    sns.scatterplot(
        x=fitted_values, y=residuals, ax=axes[0, 0], color="purple", alpha=0.7
    )
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuals vs Fitted (Linearity & Homoscedasticity Check)")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # Plot B: Normality (Q-Q Plot)
    qqplot(residuals, line="45", fit=True, ax=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q Plot (Normality Check)")

    # Plot C: Normality Distribution (Histogram)
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color="teal")
    axes[1, 0].set_title("Histogram of Residuals")
    axes[1, 0].set_xlabel("Residual Error")

    # Plot D: Independence (Residual Order Plot)
    axes[1, 1].plot(
        residuals.index, residuals, marker="o", linestyle="", color="orange", alpha=0.7
    )
    axes[1, 1].axhline(y=0, color="red", linestyle="--")
    axes[1, 1].set_title("Residuals vs Order (Independence Check)")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Residuals")

    return fig, axes


def adaptive_categorical_transformer(
    df: pd.DataFrame,
    target_type: bool,
    categorical_features=list[str],
    cardinality_threshold: int = 10,
    cv: int = 5,
):
    """
    Function to treat categorical features differently during preprocessing depending on the
    cardinality of the data. OneHotEncoding and TargetEnconding will be applied depending
    on if the caridnality is below or above a pre defined threshold.

    Args:
        df (pd.DataFrame): A pandas dataframe
        target_type (str): bool: If equals continuous, target_type within TargetEncoder
        will be explicitly set to 'continuous' to ensure correct encoding.
        categorical_features: list[str]: A list of categorical features within the
        dataset (df).
        cardinality_threshold (str): A threshold to apply different encoding depending
        on the cardinality of each categorical feature.
        cv (int): Number of folds to use in cross validation during TargetEncoding
    Returns:
        Pipeline: A pipeline that preprocesses categorical data.
    """
    df_cat = df[categorical_features].copy()
    low_card_cols, high_card_cols = [], []

    for col in df_cat.columns:
        if df_cat[col].nunique() <= cardinality_threshold:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    transformers = []
    # Low Cardinality -> One Hot Encoding
    if low_card_cols:
        transformers.append(
            (
                "one_hot",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="if_binary"
                ),
                low_card_cols,
            )
        )

    # High Cardinality -> Target Encoder
    if target_type == "continuous":
        target_encoder = TargetEncoder(cv=cv, random_state=42, target_type="continuous")
    else:
        target_encoder = TargetEncoder(
            smooth="auto", cv=cv, random_state=42
        )  # auto detect target type

    if high_card_cols:
        transformers.append(("target_enc", target_encoder, high_card_cols))

    categorical_encoder = ColumnTransformer(transformers=transformers, remainder="drop")

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            ("encoder", categorical_encoder),
        ]
    ).set_output(transform="pandas")

    return categorical_pipeline


class VIFSelector(BaseEstimator, TransformerMixin):
    """
    A Scikit-Learn compatible transformer for dropping highly collinear features using VIF.

    Variance Inflation Factor (VIF) measures how much the variance of an estimated regression 
    coefficient is increased because of collinearity. This transformer iteratively removes 
    the feature with the highest VIF score until all remaining features fall below a 
    specified threshold.

    Parameters
    ----------
    threshold : float, default=5.0
        The maximum allowable VIF score. Features with a VIF strictly greater than 
        this value will be iteratively removed. Common thresholds are 5.0 or 10.0.

    Attributes
    ----------
    keep_cols_ : list of str or list of int
        The list of feature names (or column indices) that survived the VIF filtering 
        process during the `.fit()` step.
    """
    
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.keep_cols_ = None

    def fit(self, X, y=None):
        """
        Identify which features to retain by iteratively eliminating high-VIF columns.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The training data containing numerical features to analyse for multicollinearity.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator instance.
        """
        # Convert to DataFrame to track column names easily
        df = pd.DataFrame(X)
        cols = list(df.columns)

        while True:
            if len(cols) <= 1:
                break
            # Add constant intercept required for accurate VIF
            X_with_const = add_constant(df[cols])

            # Safely compute VIF for each feature
            vifs = [
                variance_inflation_factor(X_with_const.values, i)
                for i in range(X_with_const.shape[1])
            ]

            # The first column is usually the constant, map VIF to feature names
            vif_series = (
                pd.Series(vifs[1:], index=X_with_const.columns[1:])
                if "const" in X_with_const.columns
                else pd.Series(vifs, index=X_with_const.columns)
            )

            max_vif = vif_series.max()

            if max_vif > self.threshold:
                max_col = vif_series.idxmax()
                cols.remove(max_col)
            else:
                break

        self.keep_cols_ = cols
        return self

    def transform(self, X):
        """
        Reduce the dataset to only include the features that passed the VIF threshold.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_selected_features)
            The subset of the input array containing only the non-collinear features.
        """
        df = pd.DataFrame(X)
        return df[self.keep_cols_].values

def load_pipeline_config(config_path: str) -> dict:
    """Reads and parses a YAML configuration file safely into a dictionary.

    Parameters
    ----------
    config_path : str
        The path to the YAML file on disk.

    Returns
    -------
    config : dict
        The parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
    If the file does not exist at the specified path.
    yaml.YAMLError
        If the file content contains invalid or corrupted YAML syntax.
    """
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            
            # Defensive check: if a file is blank or contains only comments, 
            # safe_load returns None and returns an empty dict instead.
            if config is None:
                return {}
                
            return config
            
        except yaml.YAMLError as e:
            # Re-raise the exception cleanly so tests can catches it
            raise yaml.YAMLError(f"Failed to parse invalid YAML file at {config_path}: {e}")
