from ._flow import (
    PrepWithReduce,
    PrepWithoutReduce,
    PrepAllNumericWithReduce,
    PrepAllNumericWithoutReduce,
)


class Preprocessing:
    
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
