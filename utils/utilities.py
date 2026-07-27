import pandas as pd

def detect_numerical_categorical_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:

    """
    Function to detect  the numerical and categorical features in a dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to detect features types

    Returns:
        tuple: A tuple of two lists, one with numerical features,
        the other categorical features.
    """

    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        numerical_features = df.dtypes[(df.dtypes != "str") &
                            (df.dtypes != object)].index.tolist()
        categorical_features = df.dtypes[(df.dtypes == "str") |
                            (df.dtypes == object)].index.tolist()

        return numerical_features, categorical_features

    except TypeError as e:
        print(f"Type validation failed: {e}, returning None")
        return None, None