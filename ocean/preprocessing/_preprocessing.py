from ._flow import (
    PrepWithReduce,
    PrepWithoutReduce,
    PrepAllNumericWithReduce,
    PrepAllNumericWithoutReduce,
)


class Preprocessing:

    """
    Preprocessing
    --------------

    Preprocessing is a class that allows you to combine several preprocessing steps such as imputer, encoder, scaler, and reduce.
    You can also set some parameters to use the preprocessing methods that suit your needs.

    Parameters
    ----------

    categoric: list, default=None
        list of categorical features in the dataset that you want to process

    impute: str, default='mean'
        method of imputing missing values using SimpleImputer. The choices available are `mean`, `median`, `most_frequent`.

    encode: str, default='onehot'
        method of encoding categorical features. The choices available are `onehot` and `ordinal`.

    scaler: str, default='minmax'
        method of scaling numerical features. The choices available are `minmax`, `standard`, and `robust`.

    reduce: bool, default=False
        use of PCA to reduce feature dimension

    Usage
    ------
    ```python
    >>> prep = Preprocessing()
    >>> X_train = prep.fit(X_train)
    ```
    """

    def __init__(
        self,
        categoric: list = None,
        impute: str = "mean",
        encode: str = "onehot",
        scaler: str = "minmax",
        reduce: bool = False,
    ):
        self.categoric = categoric
        self.impute = impute
        self.encode = encode
        self.scaler = scaler
        self.reduce = reduce

    def fit(self, X_train):
        if self.categoric is not None:
            if self.reduce:
                prep_with_reduce = PrepWithReduce(
                    X_train, self.categoric, self.impute, self.encode, self.scaler
                )
                return prep_with_reduce.fit_with_reduce()
            else:
                prep_without_reduce = PrepWithoutReduce(
                    X_train, self.categoric, self.impute, self.encode, self.scaler
                )
                return prep_without_reduce.fit_without_reduce()

        else:
            if self.reduce:
                prep_all_numeric_with_reduce = PrepAllNumericWithReduce(
                    X_train, self.impute, self.scaler
                )
                return prep_all_numeric_with_reduce.fit_with_reduce()
            else:
                prep_all_numeric_without_reduce = PrepAllNumericWithoutReduce(
                    X_train, self.impute, self.scaler
                )
                return prep_all_numeric_without_reduce.fit_without_reduce()
