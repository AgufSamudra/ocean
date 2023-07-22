from scipy.stats import loguniform
from ocean.automl._prep import scaler

# ==== REGRESSOR ====

# SVR
parameters_svr = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__kernel": ["poly", "rbf"],
    "algo__C": loguniform(1e-1, 1e1),
    "algo__epsilon": loguniform(1e-2, 5e-1),
}

# Random Forest Regressor
parameters_rfr = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_estimators": range(100, 500, 5),
    "algo__max_depth": range(1, 50),
    "algo__min_samples_split": range(2, 50),
    "algo__min_samples_leaf": range(1, 50),
    "algo__max_features": ["sqrt"],
}

# XGBoost Regressor
parameters_xgbr = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_estimators": range(100, 500, 10),
    "algo__learning_rate": loguniform(0.01, 1),
    "algo__max_depth": range(1, 10),
    "algo__subsample": loguniform(0.1, 0.9),
    "algo__colsample_bytree": loguniform(0.1, 0.9),
}

# KNN Regressor
parameters_knnr = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_neighbors": range(1, 25, 2),
    "algo__weights": ["uniform", "distance"],
    "algo__metric": ["minkowski", "euclidean", "manhattan"],
}

# ElasticNet
parameters_elastic = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__l1_ratio": loguniform(1e-4, 1),
    "algo__alpha": loguniform(1e-1, 1e1),
}


# ==== CLASSIFIER ====

# SVC
parameters_svc = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__kernel": ["rbf"],
    "algo__C": loguniform(0.01, 100),
    "algo__gamma": loguniform(0.001, 10),
    "algo__class_weight": ["balanced", None],
}

# Random Forest Classier
parameters_rfc = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_estimators": range(100, 500),
    "algo__max_depth": range(1, 50),
    "algo__min_samples_split": range(2, 50),
    "algo__min_samples_leaf": range(1, 50),
}

# XGBoost Classifier
parameters_xgbc = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_estimators": range(100, 500, 10),
    "algo__learning_rate": loguniform(0.01, 1),
    "algo__max_depth": range(1, 10),
    "algo__subsample": loguniform(0.1, 0.9),
    "algo__colsample_bytree": loguniform(0.1, 0.9),
}

# KNN Classifier
parameters_knnc = {
    "prep__numeric__scaler": scaler,
    "prep__numeric__imputer__strategy": ["mean", "median", "most_frequent"],
    "algo__n_neighbors": range(1, 20),
    "algo__weights": ["uniform", "distance"],
    "algo__metric": ["euclidean", "manhattan", "minkowski"],
}
