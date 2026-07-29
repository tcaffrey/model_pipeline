import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
