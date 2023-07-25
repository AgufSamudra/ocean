"""
save load_model by default will create a new directory called save_model
which is the model itself.
"""

# import library
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ocean.save_load_model import save_model, load_model
from sklearn.model_selection import train_test_split

# import dataset
df = pd.read_csv("[YOUR DATASET]")

# spliting dataset
X = df.drop(columns="YOUR LABELS")
y = df["YOUR LABELS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# modeling
model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

# score model before save
model.score(X_train, y_train), model.score(X_test, y_test)

# ============= SAVE AND LOAD WITH OCEAN =============

# save model
save_model("random_forest_model", model)

# load model
load_model = load_model("save_model/random_forest_model.pkl")
load_model.score(X_train, y_train), load_model.score(X_test, y_test)