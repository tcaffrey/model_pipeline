import pandas as pd
from src.utils import (
    detect_numerical_categorical_features,
    adaptive_categorical_transformer,
)
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class MachineLearningPipeline:

    def __init__(self, model_estimator):
        self.model = self._validate_model(model_estimator)
        self.preprocessor = None
        self.is_fitted = False

    @staticmethod
    def _validate_model(estimator):
        """
        Check that the provided model estimator is of a classifier or
        regressor type. For models such as XGBoost or LightGBM a check
        for fit and predict methods can also be carried out.
        """
        if isinstance(estimator, (str, list, dict, set, tuple)):
            raise TypeError(
                f"Expected an instantiated model object, but received a primitive \
                '{type(estimator).__name__}' instead. Ensure you pass an actual \
                instance like LogisticRegression() rather than text."
            )

        is_sklearn = is_classifier(estimator) or is_regressor(estimator)
        has_methods = hasattr(estimator, "fit") and hasattr(estimator, "predict")

        if not (is_sklearn or has_methods):
            raise TypeError(
                f"Invalid estimator type: '{type(estimator).__name__}'. "
                f"Must be an instantiated model with .fit() and .predict() methods."
            )
        return estimator

    def _build_preprocessor(self, df, numerical_features, categorical_features):
        """
        Internal function to build Scikit-Learn preprocessing blocks for
        numerical and categorical data.
        """

        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")

        cat_transformer = adaptive_categorical_transformer(
            df,
            regression_flag=True,
            categorical_features=categorical_features,
            cardinality_threshold=10,
            cv=5,
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, numerical_features),
                ("cat", cat_transformer, categorical_features),
            ]
        )

    def preprocess_data(self, df, is_training=True):
        """
        Detect if features in dataset are numerical or categorical and pass these
        with the dataset to be processed by the appropriate component.
        """
        df_preprocess = df.copy()
        numerical_features, categorical_features = (
            detect_numerical_categorical_features(df_preprocess)
        )

        if is_training:
            self._build_preprocessor(df, numerical_features, categorical_features)
            processed_features = self.preprocessor.fit_transform(df_preprocess)
        else:
            if not self.is_fitted:
                raise ValueError(
                    "Pipeline must be fitted before preprocessing test data."
                )
            processed_features = self.preprocessor.transform(df_preprocess)

        return processed_features

    def fit(self, X, y):
        """
        Preprocesses the features and fits the underlying model.
        """
        X_processed = self.preprocess_data(X, is_training=True)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Preprocesses new data and returns model predictions.
        """
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict(X_processed)
