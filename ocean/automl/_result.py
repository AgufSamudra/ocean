import numpy as np
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


## REGERSSOR
def _report(self):
    try:
        table_data = [
            [
                "SVR",
                "{:.2f}%".format(self.train_score_svr * 100),
                "{:.2f}%".format(self.test_score_svr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_svr))
                ),
                f"{self.time_svr:.2f}",
            ],
            [
                "KNN",
                "{:.2f}%".format(self.train_score_knnr * 100),
                "{:.2f}%".format(self.test_score_knnr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_knnr))
                ),
                f"{self.time_knnr:.2f}",
            ],
            [
                "XGBoost",
                "{:.2f}%".format(self.train_score_xgbr * 100),
                "{:.2f}%".format(self.test_score_xgbr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_xgbr))
                ),
                f"{self.time_boost:.2f}",
            ],
            [
                "ElasticNet",
                "{:.2f}%".format(self.train_score_elastic * 100),
                "{:.2f}%".format(self.test_score_elastic * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_elastic))
                ),
                f"{self.time_elastic:.2f}",
            ],
            [
                "Random Forest",
                "{:.2f}%".format(self.train_score_rfr * 100),
                "{:.2f}%".format(self.test_score_rfr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_rfr))
                ),
                f"{self.time_rfr:.2f}",
            ],
        ]

        table_headers = ["Model", "Train", "Test", "R2", "MAE", "MSE", "RMSE", "Time"]
        table_format = "fancy_grid"

        title = "Model Performance"
        title_length = len(title)
        table_width = len(table_headers) * 10 + 1

        print(" " * ((table_width - title_length) // 2) + title)
        print(tabulate(table_data, headers=table_headers, tablefmt=table_format))

    except AttributeError:
        print("Error: Please train the model first!")


def _report_tune(self):
    try:
        table_data = [
            [
                "SVR",
                "{:.2f}%".format(self.train_score_svr * 100),
                "{:.2f}%".format(self.svr_validation * 100),
                "{:.2f}%".format(self.test_score_svr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_svr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_svr))
                ),
                f"{self.time_svr:.2f}",
            ],
            [
                "KNN",
                "{:.2f}%".format(self.train_score_knnr * 100),
                "{:.2f}%".format(self.knnr_validation * 100),
                "{:.2f}%".format(self.test_score_knnr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_knnr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_knnr))
                ),
                f"{self.time_knnr:.2f}",
            ],
            [
                "XGBoost",
                "{:.2f}%".format(self.train_score_xgbr * 100),
                "{:.2f}%".format(self.boost_validation * 100),
                "{:.2f}%".format(self.test_score_xgbr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_xgbr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_xgbr))
                ),
                f"{self.time_boost:.2f}",
            ],
            [
                "ElasticNet",
                "{:.2f}%".format(self.train_score_elastic * 100),
                "{:.2f}%".format(self.elastic_validation * 100),
                "{:.2f}%".format(self.test_score_elastic * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_elastic)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_elastic))
                ),
                f"{self.time_elastic:.2f}",
            ],
            [
                "Random Forest",
                "{:.2f}%".format(self.train_score_rfr * 100),
                "{:.2f}%".format(self.rfr_validation * 100),
                "{:.2f}%".format(self.test_score_rfr * 100),
                "{:.4f}".format(r2_score(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(mean_absolute_error(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(mean_squared_error(self.y_test, self.y_pred_rfr)),
                "{:.2f}".format(
                    np.sqrt(mean_squared_error(self.y_test, self.y_pred_rfr))
                ),
                f"{self.time_rfr:.2f}",
            ],
        ]

        table_headers = [
            "Model",
            "Train",
            "Validation",
            "Test",
            "R2",
            "MAE",
            "MSE",
            "RMSE",
            "Time",
        ]
        table_format = "fancy_grid"

        title = "Model Performance"
        title_length = len(title)
        table_width = len(table_headers) * 10 + 1

        print(" " * ((table_width - title_length) // 2) + title)
        print(tabulate(table_data, headers=table_headers, tablefmt=table_format))

    except AttributeError:
        print("Error: Please train the model first!")


def _best_model(self):
    scores = {
        "SVR": self.test_score_svr,
        "Random Forest": self.test_score_rfr,
        "XGBoost": self.test_score_xgbr,
        "K-Nearest Neighbour": self.test_score_knnr,
        "ElasticNet": self.test_score_elastic,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_algorithm, best_score = sorted_scores[0]

    print("Best Model:", best_algorithm)


def _best_model_tune(self):
    scores = {
        "SVR": self.test_score_svr,
        "Random Forest": self.test_score_rfr,
        "XGBoost": self.test_score_xgbr,
        "K-Nearest Neighbour": self.test_score_knnr,
        "ElasticNet": self.test_score_elastic,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_algorithm, best_score = sorted_scores[0]
    best_params = None

    if best_algorithm == "SVR":
        best_params = self.model_svr_params
    elif best_algorithm == "Random Forest":
        best_params = self.model_rfr_params
    elif best_algorithm == "XGBoost":
        best_params = self.model_boost_params
    elif best_algorithm == "K-Nearest Neighbour":
        best_params = self.model_knn_params
    elif best_algorithm == "ElasticNet":
        best_params = self.model_elastic_params

    print("Best Model:", best_algorithm)
    print("Params:", best_params)


## CLASSIFIER
def _report_classifier(self):
    try:
        table_data = [
            [
                "SVC",
                "{:.2f}%".format(self.train_score_svc * 100),
                "{:.2f}%".format(self.test_score_svc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_svc)),
                f"{self.time_svc:.2f}",
            ],
            [
                "KNN",
                "{:.2f}%".format(self.train_score_knnc * 100),
                "{:.2f}%".format(self.test_score_knnc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_knnc)),
                f"{self.time_knnc:.2f}",
            ],
            [
                "XGBoost",
                "{:.2f}%".format(self.train_score_xgbc * 100),
                "{:.2f}%".format(self.test_score_xgbc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_xgbc)),
                f"{self.time_boost:.2f}",
            ],
            [
                "Random Forest",
                "{:.2f}%".format(self.train_score_rfc * 100),
                "{:.2f}%".format(self.test_score_rfc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_rfc)),
                f"{self.time_rfc:.2f}",
            ],
        ]

        table_headers = [
            "Model",
            "Train",
            "Test",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Time",
        ]
        table_format = "fancy_grid"

        title = "Model Performance"
        title_length = len(title)
        table_width = len(table_headers) * 10 + 1

        print(" " * ((table_width - title_length) // 2) + title)
        print(tabulate(table_data, headers=table_headers, tablefmt=table_format))

    except AttributeError:
        print("Error: Please train the model first!")


def _report_classifier_tune(self):
    try:
        table_data = [
            [
                "SVC",
                "{:.2f}%".format(self.train_score_svc * 100),
                "{:.2f}%".format(self.svc_validation * 100),
                "{:.2f}%".format(self.test_score_svc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_svc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_svc)),
                f"{self.time_svc:.2f}",
            ],
            [
                "KNN",
                "{:.2f}%".format(self.train_score_knnc * 100),
                "{:.2f}%".format(self.knnc_validation * 100),
                "{:.2f}%".format(self.test_score_knnc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_knnc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_knnc)),
                f"{self.time_knnc:.2f}",
            ],
            [
                "XGBoost",
                "{:.2f}%".format(self.train_score_xgbc * 100),
                "{:.2f}%".format(self.boost_classification_validation * 100),
                "{:.2f}%".format(self.test_score_xgbc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_xgbc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_xgbc)),
                f"{self.time_boost:.2f}",
            ],
            [
                "Random Forest",
                "{:.2f}%".format(self.train_score_rfc * 100),
                "{:.2f}%".format(self.rfc_validation * 100),
                "{:.2f}%".format(self.test_score_rfc * 100),
                "{:.4f}".format(accuracy_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(precision_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(recall_score(self.y_test, self.y_pred_rfc)),
                "{:.2f}".format(f1_score(self.y_test, self.y_pred_rfc)),
                f"{self.time_rfc:.2f}",
            ],
        ]

        table_headers = [
            "Model",
            "Train",
            "Validation",
            "Test",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Time",
        ]
        table_format = "fancy_grid"

        title = "Model Performance"
        title_length = len(title)
        table_width = len(table_headers) * 10 + 1

        print(" " * ((table_width - title_length) // 2) + title)
        print(tabulate(table_data, headers=table_headers, tablefmt=table_format))

    except AttributeError:
        print("Error: Please train the model first!")


def _best_model_classifier(self):
    scores = {
        "SVC": self.test_score_svc,
        "Random Forest": self.test_score_rfc,
        "XGBoost": self.test_score_xgbc,
        "K-Nearest Neighbour": self.test_score_knnc,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_algorithm, best_score = sorted_scores[0]

    print("Best Model:", best_algorithm)


def _best_model_tune_classifier(self):
    scores = {
        "SVC": self.test_score_svc,
        "Random Forest": self.test_score_rfc,
        "XGBoost": self.test_score_xgbc,
        "K-Nearest Neighbour": self.test_score_knnc,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_algorithm, best_score = sorted_scores[0]
    best_params = None

    if best_algorithm == "SVC":
        best_params = self.model_svc_params
    elif best_algorithm == "Random Forest":
        best_params = self.model_rfc_params
    elif best_algorithm == "XGBoost":
        best_params = self.model_boost_params
    elif best_algorithm == "K-Nearest Neighbour":
        best_params = self.model_knn_params

    print("Best Model:", best_algorithm)
    print("Params:", best_params)
