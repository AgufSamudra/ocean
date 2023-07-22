from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

scaler = [StandardScaler(), MinMaxScaler(), RobustScaler()]


def _prep(numeric, categoric):
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer()), ("scaler", RobustScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numerical_pipeline, numeric),
            ("categoric", categorical_pipeline, categoric),
        ]
    )

    return preprocessor


def _prep_all_numeric(numeric):
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer()), ("scaler", RobustScaler())]
    )

    preprocessor = ColumnTransformer([("numeric", numerical_pipeline, numeric)])

    return preprocessor
