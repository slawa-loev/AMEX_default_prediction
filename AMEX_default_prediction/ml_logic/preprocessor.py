from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd

from colorama import Fore, Style

from AMEX_default_prediction.ml_logic.params import CAT_VARS

def preprocess_features(X: pd.DataFrame) -> np.ndarray:


    num_vars = [feature for feature in X if feature not in CAT_VARS + ["customer_ID", "S_2"]] ## exclude dates and IDs (first two columns)
    #str_vars = [feature for feature in X[2:] if not pd.api.types.is_numeric_dtype(X_red[feature])] ## columns that are not numeric at all
    #red_cat_vars = [feature for feature in cat_vars if feature not in dropped_columns + str_vars] ## remaining categorical variables that have no string values
    cat_vars = [feature for feature in X.columns if feature in CAT_VARS] ## remaining categorical variables

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Create a scikit-learn preprocessor
        that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of different fixed shape (_, 65)
        """
        # impute mean/most frequent value for other nans (specific to group?)
        # robustscale all numerical values

        num_imputer = SimpleImputer(strategy='mean')
        num_scaler = RobustScaler()

        #num_imputer = KNNImputer(n_neighbors=2) ## KNNIMputer is computationally demanding
        ## should come AFTER SCALING

        num_pipe = make_pipeline(num_imputer, num_scaler)

        #str_trans = OrdinalEncoder() # is only needed if one wants to do knnimputer

        #nan_trans = FunctionTransformer(nan_imp)

        #nan_trans = FunctionTransformer(lambda X: X.applymap(lambda x: np.nan if x in [-1,-1.0, "-1.0", "-1"] else x))

        cat_imputer = SimpleImputer(strategy="most_frequent") ## replace with KNNimputer on one neighbour, after transforming to numericals
        #cat_imputer = KNNImputer(n_neighbors=1) # introducing it did not improve performance, but is computationally demanding
        cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') ## what happens to the old columns?
        cat_pipe = make_pipeline(cat_imputer, cat_encoder)
        #str_pipe = make_pipeline(cat_imputer, str_trans, cat_encoder)
        #str_pipe = make_pipeline(cat_imputer, cat_encoder)

        final_preprocessor= ColumnTransformer([
                ('num_pip', num_pipe, num_vars),
                ('cat_pip', cat_pipe, cat_vars)],
                remainder='drop' ## all columns not in num_vars and red_cat_vars are dropped.
            )

        return final_preprocessor

    print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed
