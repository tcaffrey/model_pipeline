import pandas as pd
from src.utils import (
    detect_numerical_categorical_features,
    adaptive_categorical_transformer,
    load_pipeline_config,
    VIFSelector,
    CorrelationFilter,
    is_pipeline_empty,
)
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

class MachineLearningPipeline:

    def __init__(self, config: dict, model_estimator):
        self.model = self._validate_model(model_estimator)
        # self.VIFSelector = VIFSelector(threshold=5.0)
        # self.preprocessor = None
        self.is_fitted = False
        self.config = config
        self.target_type = self.config.get("target_type", "")
        self.preprocessing_steps = self.config.get("preprocessing_pipeline", [])
        self.candidates = self.config.get("model_candidates", [])

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

    def _build_column_transformer(self, df, y, numerical_features, categorical_features):
        """
        Internal function to build Scikit-Learn preprocessing blocks for
        numerical and categorical data.
        """
        numerical_transformer_steps = []
        categorical_transformer = []

        for step in self.preprocessing_steps:
            name = step.get("name")
            if not step.get("enabled", False):
                continue

            if name == "handle_missing_data":
                strategy = step.get("params", {}).get("strategy", "mean")
                numerical_transformer_steps.append((name, SimpleImputer(strategy=strategy)))

            if name == "scale_features":
                method = step.get("params", {}).get("method", "standard")
                scaler = StandardScaler() if method == "standard" else MinMaxScaler()
                numerical_transformer_steps.append((name, scaler))

            if name == "remove_high_correlation":
                corr_threshold = step.get("params", {}).get("threshold", 0.85)
                numerical_transformer_steps.append((name, CorrelationFilter(threshold=corr_threshold)))

            if name == "categorical_encoding":
                categorical_transformer = adaptive_categorical_transformer(
                    df,
                    target_type=self.target_type,
                    categorical_features=categorical_features,
                    cardinality_threshold=10,
                    cv=5,
                )

        numerical_transformer = Pipeline(numerical_transformer_steps).set_output(transform="pandas")

        numerical_pipeline_check = is_pipeline_empty(numerical_transformer)
        categorical_pipeline_check = is_pipeline_empty(categorical_transformer)

        if numerical_pipeline_check:
            numerical_transformer = "passthrough"
        if categorical_pipeline_check:
            categorical_transformer = "passthrough"

        print(categorical_pipeline_check)
        print(categorical_transformer, categorical_features)

        column_transformer = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return column_transformer

    def _build_preprocessor(self, df, y, is_training=True):
        """
        Detect if features in dataset are numerical or categorical and pass these
        with the dataset to be processed by the appropriate component.
        """
        df_preprocess = df.copy()
        preprocessor_pipeline = []

        numerical_features, categorical_features = (
            detect_numerical_categorical_features(df_preprocess)
        )
        column_transformer = self._build_column_transformer(df, y, numerical_features, categorical_features)

        if is_training:
            preprocessor_pipeline.append(('column_transformer', column_transformer))
            for step in self.preprocessing_steps:
                name = step.get("name")
                if not step.get("enabled", False):
                    continue
                if name == "remove_high_mulitcollinearity":
                    vif_threshold = step.get("params", {}).get("vif_threshold", 5.0)
                    preprocessor_pipeline.append((name, VIFSelector(threshold=vif_threshold)))
                        
            return Pipeline(preprocessor_pipeline)
            # processed_features = self.preprocessor.fit_transform(df_preprocess, y)
        # else:
        #     if not self.is_fitted:
        #         raise ValueError(
        #             "Pipeline must be fitted before preprocessing test data."
        #         )
        #     processed_features = self.preprocessor.transform(df_preprocess)

        # return processed_features

    def fit(self, X, y):
        """
        Preprocesses the features and fits the underlying model.
        """
        self.preprocessor = self._build_preprocessor(X, y, is_training=True)
        X_processed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_processed, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Preprocesses new data and returns model predictions.
        """
        # X_processed = self.preprocess_data(X, y=None, is_training=False)
        if not self.is_fitted:
            raise ValueError(
                "Pipeline must be fitted before preprocessing test data."
            )
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def tune_single_model(self, param_grid, X, y, cv=5, scoring='accuracy'):
        """
        Runs GridSearchCV for a single model object.
        """
        if self.model is None:
            raise ValueError("Cannot tune model: self.model estimator is not initialized.")
        
        # numerical_features, categorical_features = (
        #     detect_numerical_categorical_features(X_train)
        # )

        # column_transformer = self._build_column_transformer(X_train, y_train, numerical_features, categorical_features)
        pipeline = self._build_preprocessor(X, y, is_training=True)
        # pipeline = Pipeline(self.preprocessor)
        pipeline.steps.append(("model", self.model))
         
        # 1. Dynamically build the pipeline
        # pipeline = Pipeline([
        #     ('preprocessing', column_transformer),
        #     ('vif_filter', VIFSelector(5.0)),
        #     ('model', self.model)
        # ])
        
        # 2. Map raw keys to pipeline step syntax ('clf__parameter')
        formatted_params = {f"model__{k}": v for k, v in param_grid.items()}
        
        # 3. Initialize and run GridSearch
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=formatted_params,
            cv=cv,
            scoring=scoring,
            error_score='raise',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Returns the completed GridSearchCV object containing results and the best estimator
        return grid_search

if __name__ == "__main__":
    config_dict = load_pipeline_config("pipeline_config.yaml")
    linear_model = LinearRegression()
    MLPipeline = MachineLearningPipeline(config = config_dict, model_estimator = linear_model)