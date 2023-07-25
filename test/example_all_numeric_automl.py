"""
If all datasets contain numeric features, then there is no need to enter parameters
'categoric' because ocean will read all features are numeric otherwise
fill in the 'categoric' parameter. So simply forget those parameters.
"""

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
model = AutoMLClassifier(X, y)
model.fit()

# =========== CLASSIFIER WITH TUNING PARAMS ===========
model = AutoMLClassifier(X, y, tune_params=True)
model.fit()


# =========== REGRESSOR WITHOUT TUNING PARAMS ===========
model = AutoMLRegressor(X, y)
model.fit()

# =========== REGRESSOR WITH TUNING PARAMS ===========
model = AutoMLRegressor(X, y, tune_params=True)
model.fit()