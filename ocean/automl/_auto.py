import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from ocean.automl._params import *
from ocean.automl._prep import _prep, _prep_all_numeric
from ocean.automl._result import (
    _report,
    _report_tune,
    _best_model,
    _best_model_tune,
    _report_classifier,
    _report_classifier_tune,
    _best_model_classifier,
    _best_model_tune_classifier,
)


##### Auto ML Regressor #####
class AutoMLRegressor(object):
    """
    AutoMLRegressor
    ---------------

    Class for finding a base model as an initial benchmark without performing tuning.
    Users can directly input the train and test data into this class.

    Parameters
    ----------
    X
        The input features for the regression task.

    y
        The target variable for the regression task.

    categoric: list
        A list of names of the categorical features in the data. This process will be used to separate categorical and numeric features in preprocessing.

    test_size: float, default=0.2
        The proportion of data to be allocated to the test set. Specify how much data you want to allocate to the test set.

    tune_params: bool, default=False
        If True, AutoML will perform parameter tuning that has been determined by the package.
        `Warning`: This process will take some time, so consider it if you have large data.

    nume_impute: default='mean'
        The process of filling in missing data in the dataset. There are several options to choose from: `mean`, `median`, `most_frequent`.

    n_iter: int, default=20
        The process of combination if `tune_params=True`. How many rounds of combination do you want to do.

    cv: int, default=3
        Cross validation on the training data, you can enter more than this.
        `Warning`: Entering more values for cv will affect the accuracy calculation.

    Usage
    -----
    ```python
    >>> from ocean.automl import AutoMLRegressor
    >>> auto_ml = AutoMLRegressor(X, y, categoric)
    >>> auto_ml.fit()
    ```
    """

    def __init__(
        self,
        X,
        y,
        categoric: list = None,
        test_size=0.2,
        tune_params=False,
        n_iter=50,
        cv=3,
    ):
        self.X = X
        self.y = y
        self.categoric = categoric
        if self.categoric != None:
            self.numeric = list(set(self.X).difference(self.categoric))
        self.test_size = test_size
        self.n_iter = n_iter
        self.cv = cv
        self.tune_params = tune_params

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, shuffle=True, random_state=42
        )

    def _svr(self):
        self.start_time_svr = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", SVR()),
                ]
            )
        else:
            pipeline = Pipeline(
                [("prep", _prep_all_numeric(numeric=self.X.columns)), ("algo", SVR())]
            )

        model_svr = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_svr = RandomizedSearchCV(
                pipeline,
                parameters_svr,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_svr.fit(self.X_train, self.y_train)
            self.model_svr_params = model_svr.best_params_
            self.svr_validation = model_svr.best_score_
        else:
            model_svr.fit(self.X_train, self.y_train)

        self.stop_time_svr = time.time()
        self.time_svr = self.stop_time_svr - self.start_time_svr

        self.train_score_svr = model_svr.score(self.X_train, self.y_train)
        self.test_score_svr = model_svr.score(self.X_test, self.y_test)

        self.y_pred_svr = model_svr.predict(self.X_test)

    def _forest(self):
        self.start_time_rfr = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", RandomForestRegressor()),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", RandomForestRegressor()),
                ]
            )

        model_rfr = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_rfr = RandomizedSearchCV(
                pipeline,
                parameters_rfr,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_rfr.fit(self.X_train, self.y_train)
            self.model_rfr_params = model_rfr.best_params_
            self.rfr_validation = model_rfr.best_score_
        else:
            model_rfr.fit(self.X_train, self.y_train)

        self.stop_time_rfr = time.time()
        self.time_rfr = self.stop_time_rfr - self.start_time_rfr

        self.train_score_rfr = model_rfr.score(self.X_train, self.y_train)
        self.test_score_rfr = model_rfr.score(self.X_test, self.y_test)

        self.y_pred_rfr = model_rfr.predict(self.X_test)

    def _boost(self):
        self.start_time_boost = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", XGBRegressor()),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", XGBRegressor()),
                ]
            )

        model_boost = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_boost = RandomizedSearchCV(
                pipeline,
                parameters_xgbr,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_boost.fit(self.X_train, self.y_train)
            self.model_boost_params = model_boost.best_params_
            self.boost_validation = model_boost.best_score_
        else:
            model_boost.fit(self.X_train, self.y_train)

        self.stop_time_boost = time.time()
        self.time_boost = self.stop_time_boost - self.start_time_boost

        self.train_score_xgbr = model_boost.score(self.X_train, self.y_train)
        self.test_score_xgbr = model_boost.score(self.X_test, self.y_test)

        self.y_pred_xgbr = model_boost.predict(self.X_test)

    def _knn(self):
        self.start_time_knnr = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", KNeighborsRegressor()),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", KNeighborsRegressor()),
                ]
            )

        model_knn = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_knn = RandomizedSearchCV(
                pipeline,
                parameters_knnr,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_knn.fit(self.X_train, self.y_train)
            self.model_knn_params = model_knn.best_params_
            self.knnr_validation = model_knn.best_score_
        else:
            model_knn.fit(self.X_train, self.y_train)

        self.stop_time_knnr = time.time()
        self.time_knnr = self.stop_time_knnr - self.start_time_knnr

        self.train_score_knnr = model_knn.score(self.X_train, self.y_train)
        self.test_score_knnr = model_knn.score(self.X_test, self.y_test)

        self.y_pred_knnr = model_knn.predict(self.X_test)

    def _elastic(self):
        self.start_time_elastic = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", ElasticNet()),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", ElasticNet()),
                ]
            )

        model_elstic = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_elstic = RandomizedSearchCV(
                pipeline,
                parameters_elastic,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_elstic.fit(self.X_train, self.y_train)
            self.model_elastic_params = model_elstic.best_params_
            self.elastic_validation = model_elstic.best_score_
        else:
            model_elstic.fit(self.X_train, self.y_train)

        self.stop_time_elastic = time.time()
        self.time_elastic = self.stop_time_elastic - self.start_time_elastic

        self.train_score_elastic = model_elstic.score(self.X_train, self.y_train)
        self.test_score_elastic = model_elstic.score(self.X_test, self.y_test)

        self.y_pred_elastic = model_elstic.predict(self.X_test)

    def fit(self):
        self.start_time = time.time()

        self._boost()
        self._forest()
        self._svr()
        self._knn()
        self._elastic()

        self.stop_time = time.time()
        self.time_all = self.stop_time - self.start_time
        if self.tune_params != False:
            print(f"Time: {self.time_all:.2f} Second")
            print("Mode: WITH PARAMS")
            _report_tune(self)
            _best_model_tune(self)
        else:
            print(f"Time: {self.time_all:.2f} Second")
            print("Mode: NOT WITH PARAMS")
            _report(self)
            _best_model(self)


##### Auto ML Classifier #####
class AutoMLClassifier(object):
    """
    AutoMLClassifier
    ---------------

    Class for finding a base model as an initial benchmark without performing tuning.
    Users can directly input the train and test data into this class.

    Parameters
    ----------
    X
        The input features for the regression task.

    y
        The target variable for the regression task.

    categoric: list
        A list of names of the categorical features in the data. This process will be used to separate categorical and numeric features in preprocessing.

    test_size: float, default=0.2
        The proportion of data to be allocated to the test set. Specify how much data you want to allocate to the test set.

    tune_params: bool, default=False
        If True, AutoML will perform parameter tuning that has been determined by the package.
        `Warning`: This process will take some time, so consider it if you have large data.

    nume_impute: default='mean'
        The process of filling in missing data in the dataset. There are several options to choose from: `mean`, `median`, `most_frequent`.

    n_iter: int, default=20
        The process of combination if `tune_params=True`. How many rounds of combination do you want to do.

    cv: int, default=3
        Cross validation on the training data, you can enter more than this.
        `Warning`: Entering more values for cv will affect the accuracy calculation.

    Usage
    -----
    ```python
    >>> from ocean.automl import AutoMLClassifier
    >>> auto_ml = AutoMLClassifier(X, y, categoric)
    >>> auto_ml.fit()
    ```
    """

    def __init__(
        self,
        X,
        y,
        categoric: list = None,
        test_size=0.2,
        tune_params=False,
        n_iter=50,
        cv=3,
    ):
        self.X = X
        self.y = y
        self.categoric = categoric
        if self.categoric != None:
            self.numeric = list(set(self.X).difference(self.categoric))
        self.test_size = test_size
        self.n_iter = n_iter
        self.cv = cv
        self.tune_params = tune_params

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, stratify=self.y, random_state=42
        )

    def _svc(self):
        self.start_time_svc = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", SVC()),
                ]
            )
        else:
            pipeline = Pipeline(
                [("prep", _prep_all_numeric(numeric=self.X.columns)), ("algo", SVC())]
            )

        model_svc = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_svc = RandomizedSearchCV(
                pipeline,
                parameters_svc,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_svc.fit(self.X_train, self.y_train)
            self.model_svc_params = model_svc.best_params_
            self.svc_validation = model_svc.best_score_
        else:
            model_svc.fit(self.X_train, self.y_train)

        self.stop_time_svc = time.time()
        self.time_svc = self.stop_time_svc - self.start_time_svc

        self.train_score_svc = model_svc.score(self.X_train, self.y_train)
        self.test_score_svc = model_svc.score(self.X_test, self.y_test)

        self.y_pred_svc = model_svc.predict(self.X_test)

    def _forest_classifier(self):
        self.start_time_rfc = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", RandomForestClassifier(n_jobs=-1, random_state=42)),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", RandomForestClassifier(n_jobs=-1, random_state=42)),
                ]
            )

        model_rfc = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_rfc = RandomizedSearchCV(
                pipeline,
                parameters_rfc,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_rfc.fit(self.X_train, self.y_train)
            self.model_rfc_params = model_rfc.best_params_
            self.rfc_validation = model_rfc.best_score_
        else:
            model_rfc.fit(self.X_train, self.y_train)

        self.stop_time_rfc = time.time()
        self.time_rfc = self.stop_time_rfc - self.start_time_rfc

        self.train_score_rfc = model_rfc.score(self.X_train, self.y_train)
        self.test_score_rfc = model_rfc.score(self.X_test, self.y_test)

        self.y_pred_rfc = model_rfc.predict(self.X_test)

    def _boost_classifier(self):
        self.start_time_boost = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", XGBClassifier(n_jobs=-1, random_state=42)),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", XGBClassifier(n_jobs=-1, random_state=42)),
                ]
            )

        model_boost = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_boost = RandomizedSearchCV(
                pipeline,
                parameters_xgbc,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_boost.fit(self.X_train, self.y_train)
            self.model_boost_params = model_boost.best_params_
            self.boost_classification_validation = model_boost.best_score_
        else:
            model_boost.fit(self.X_train, self.y_train)

        self.stop_time_boost = time.time()
        self.time_boost = self.stop_time_boost - self.start_time_boost

        self.train_score_xgbc = model_boost.score(self.X_train, self.y_train)
        self.test_score_xgbc = model_boost.score(self.X_test, self.y_test)

        self.y_pred_xgbc = model_boost.predict(self.X_test)

    def _knn_classifier(self):
        self.start_time_knnc = time.time()
        if self.categoric != None:
            pipeline = Pipeline(
                [
                    ("prep", _prep(numeric=self.numeric, categoric=self.categoric)),
                    ("algo", KNeighborsClassifier(n_jobs=-1)),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("prep", _prep_all_numeric(numeric=self.X.columns)),
                    ("algo", KNeighborsClassifier(n_jobs=-1)),
                ]
            )

        model_knn = pipeline

        # jika tune = True
        if self.tune_params != False:
            model_knn = RandomizedSearchCV(
                pipeline,
                parameters_knnc,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            model_knn.fit(self.X_train, self.y_train)
            self.model_knn_params = model_knn.best_params_
            self.knnc_validation = model_knn.best_score_
        else:
            model_knn.fit(self.X_train, self.y_train)

        self.stop_time_knnc = time.time()
        self.time_knnc = self.stop_time_knnc - self.start_time_knnc

        self.train_score_knnc = model_knn.score(self.X_train, self.y_train)
        self.test_score_knnc = model_knn.score(self.X_test, self.y_test)

        self.y_pred_knnc = model_knn.predict(self.X_test)

    def fit(self):
        self.start_time = time.time()

        self._boost_classifier()
        self._forest_classifier()
        self._knn_classifier()
        self._svc()

        self.stop_time = time.time()
        self.time_all = self.stop_time - self.start_time
        if self.tune_params != False:
            print(f"Time: {self.time_all:.2f} Second")
            print("Mode: WITH PARAMS")
            _report_classifier_tune(self)
            _best_model_tune_classifier(self)
        else:
            print(f"Time: {self.time_all:.2f} Second")
            print("Mode: NOT WITH PARAMS")
            _report_classifier(self)
            _best_model_classifier(self)
