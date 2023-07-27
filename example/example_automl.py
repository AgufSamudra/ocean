# import your package
import pandas as pd
from ocean.automl import AutoMLRegressor, AutoMLClassifier

# read csv file
df = pd.read_csv("[YOUR DATASET]")

# split x and y data
X = df.drop(columns="YOUR LABELS")
y = df["YOUR LABELS"]

# choice your categoric feature
categoric = ["[LIST OF CATEGORIC FEATURE]"]


# =========== CLASSIFIER WITHOUT TUNING PARAMS ===========
model = AutoMLClassifier(X, y, categoric=categoric)
model.fit()

# =========== CLASSIFIER WITH TUNING PARAMS ===========
model = AutoMLClassifier(X, y, categoric=categoric, tune_params=True)
model.fit()


# =========== REGRESSOR WITHOUT TUNING PARAMS ===========
model = AutoMLRegressor(X, y, categoric=categoric)
model.fit()

# =========== REGRESSOR WITH TUNING PARAMS ===========
model = AutoMLRegressor(X, y, categoric=categoric, tune_params=True)
model.fit()