import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.api import qqplot


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
    regression_flag: bool,
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
        regression_flag (str): bool: If set to true, target_type within TargetEncoder
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
    if regression_flag:
        target_encoder = TargetEncoder(cv=cv, random_state=42, target_type="continuous")
    else:
        target_encoder = TargetEncoder(
            cv=cv, random_state=42
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
