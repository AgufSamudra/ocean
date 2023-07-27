"""
Preprocessing is a specially designed module to make things easier
You are preprocessing the dataset that you have. With this module,
You can save time and effort preparing your data, without compromising
quality and flexibility
"""

import pandas as pd
from ocean.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

# read csv file
df = pd.read_csv("[YOUR DATASET]")

# split x and y data
X = df.drop(columns="YOUR LABELS")
y = df["YOUR LABEL"]

# choice your categoric feature
categoric = ["[LIST OF CATEGORIC FEATURE]"]

# split train and test
X_train, X_test, t_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""
You don't need to worry about parameters, because we have already defined their default values for you.
You only need to write short and simple code for parameters. However, if you want to change
or adjust the parameters according to your wishes, you can see the class docstring that we provide.
There, you will find complete and easy instructions for setting parameters.
"""

# =============== WITH REDUCE ===============
prep = Preprocessing(categoric, reduce=True)
X_train = prep.fit(X_train)

# =============== WITHOUT REDUCE ===============
prep = Preprocessing(categoric)
X_train = prep.fit(X_train)


# =============== ALL NUMERIC FEATURE WITH REDUCE ===============
prep = Preprocessing(reduce=True)
X_train = prep.fit(X_train)

# =============== ALL NUMERIC FEATURE WITHOUT REDUCE ===============
prep = Preprocessing()
X_train = prep.fit(X_train)
