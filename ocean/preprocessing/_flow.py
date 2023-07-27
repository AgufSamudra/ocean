from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.decomposition import PCA

encoder = {"onehot": OneHotEncoder(), "ordinal": OrdinalEncoder()}

scaler = {
    "minmax": MinMaxScaler(),
    "standard": StandardScaler(),
    "robust": RobustScaler(),
}


class PrepWithoutReduce:
    
    def __init__(self, X_train, categoric, impute, encode, scaler):
        self.X_train = X_train
        self.categoric = categoric
        self.impute = impute
        self.encode = encode
        self.scaler = scaler

    def fit_without_reduce(self):
        self.numeric = list(set(self.X_train).difference(self.categoric))

        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy=f"{self.impute}")),
                ("scaler", scaler[self.scaler]),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", encoder[self.encode]),
            ]
        )

        preprocessing = ColumnTransformer(
            [
                ("numeric", numerical_pipeline, self.numeric),
                ("categoric", categorical_pipeline, self.categoric),
            ]
        )

        prep = preprocessing.fit_transform(self.X_train)
        columns = preprocessing.get_feature_names_out()

        df = pd.DataFrame(prep, columns=columns)

        return df


class PrepWithReduce:
    
    def __init__(self, X_train, categoric, impute, encode, scaler):
        self.X_train = X_train
        self.categoric = categoric
        self.impute = impute
        self.encode = encode
        self.scaler = scaler

    def fit_with_reduce(self):
        self.numeric = list(set(self.X_train).difference(self.categoric))

        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy=f"{self.impute}")),
                ("scaler", scaler[self.scaler]),
                ("pca", PCA(n_components="mle", whiten=True, random_state=42)),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", encoder[self.encode]),
            ]
        )

        preprocessing = ColumnTransformer(
            [
                ("numeric", numerical_pipeline, self.numeric),
                ("categoric", categorical_pipeline, self.categoric),
            ]
        )

        prep = preprocessing.fit_transform(self.X_train)
        columns = preprocessing.get_feature_names_out()

        df = pd.DataFrame(prep, columns=columns)

        return df


class PrepAllNumericWithoutReduce:
    
    def __init__(self, X_train, impute, scaler):
        self.X_train = X_train
        self.impute = impute
        self.scaler = scaler

    def fit_without_reduce(self):
        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy=f"{self.impute}")),
                ("scaler", scaler[self.scaler]),
            ]
        )

        preprocessing = ColumnTransformer(
            [("numeric", numerical_pipeline, self.X_train.columns)]
        )

        prep = preprocessing.fit_transform(self.X_train)
        columns = preprocessing.get_feature_names_out()

        df = pd.DataFrame(prep, columns=columns)

        return df


class PrepAllNumericWithReduce:
    
    def __init__(self, X_train, impute, scaler):
        self.X_train = X_train
        self.impute = impute
        self.scaler = scaler

    def fit_with_reduce(self):
        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy=f"{self.impute}")),
                ("scaler", scaler[self.scaler]),
                ("pca", PCA(n_components="mle", whiten=True, random_state=42)),
            ]
        )

        preprocessing = ColumnTransformer(
            [("numeric", numerical_pipeline, self.X_train.columns)]
        )

        prep = preprocessing.fit_transform(self.X_train)
        columns = preprocessing.get_feature_names_out()

        df = pd.DataFrame(prep, columns=columns)

        return df
