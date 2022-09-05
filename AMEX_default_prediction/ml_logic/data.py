from curses.ascii import alt
from AMEX_default_prediction.ml_logic.params import (COLUMN_NAMES_RAW,
                                            DTYPES_RAW_OPTIMIZED,
                                            DTYPES_RAW_OPTIMIZED_HEADLESS,
                                            DTYPES_PROCESSED_OPTIMIZED
                                            )

from AMEX_default_prediction.data_sources.local_disk import (get_pandas_chunk, save_local_chunk)

from AMEX_default_prediction.data_sources.big_query import (get_bq_chunk, save_bq_chunk)

import os
import pandas as pd

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import itertools

import gc

from AMEX_default_prediction.ml_logic.params import CAT_VARS

# TODO: build aggregate function?

# ====================================================
# Get the difference
# ====================================================
def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1


def get_feature_data():

    #return pd.read_csv('/content/data/train_data.csv')

    pass

def get_label_data():
    # return pd.read_csv('/content/data/train_labels.csv')
    pass

## DATA AGGREGATION AND FEATURE ENGINEERING ##
## TODO:
# 1. build preprocessor that works well with this
# 2. add other stpes from https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
# 3. Add perhaps some feature engineering from https://www.kaggle.com/code/swimmy/tuffline-amex-anotherfeaturelgbm
def data_agg(): ## pass raw data

    df = get_feature_data()

    features = df.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list() # get all feature names, except customer_ID and dates

    ## numerical features aggregation
    num_features = [feature for feature in features if feature not in CAT_VARS] ## get num features

    train_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last']) # give summary statistics for each numerical feature
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns] # join the column name tuples to a single name
    train_num_agg.reset_index(inplace = True) # get the customer_ID in as a column again and reset index

    ## get lag differnece data

    train_diff = get_difference(df, num_features)

    ## categorical feature aggregation
    train_cat_agg = df.groupby("customer_ID")[cat_vars].agg(['count', 'last', 'nunique']) # give summary statistics for each categrocial feature
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns] # join the column name tuples to a single name
    train_cat_agg.reset_index(inplace = True) # get the customer_ID in as a column again and reset index

    ## get label data

    train_labels = get_label_data()

    ## merge dfs

    df_agg = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')

    del train_num_agg, train_cat_agg, train_diff, train_labels
    gc.collect()

    return df_agg


def alt_nan_imp(X):

    X_imp = X.copy()

    alt_nan_list = [-1,-1.0, "-1.0", "-1"]

    cat_columns = [column for column in X_imp.columns if column in CAT_VARS]

    X_imp[cat_columns] = X_imp[cat_columns].applymap(lambda x: np.nan if x in alt_nan_list else x)

    return X_imp

def clean_data(df: pd.DataFrame, corr_cutoff=0.95, nan_cutoff=0.8) -> pd.DataFrame:
    """
    perform the following data cleaning:
    1. Adds proper NaN values for values in the data that could be "alternative NaNs" such as -1
    2. Drops columns with more NaNs than nan_cutoff
    3. Drops columns if they correlate >= corr_cutoff with others.

    Returns a cleaned dataframe and a list of removed column names.
    """

    df_red = alt_nan_imp(df)

    df_red = df_red.dropna(axis=1, thresh=int((1-nan_cutoff)*len(df))) # dropping columns with too many nans

    df_red_corr = df_red.corr()
    df_red_corr = df_red_corr.unstack().reset_index() # Unstack correlation matrix
    df_red_corr.columns = ['feature_1','feature_2', 'correlation_all'] # rename columns
    df_red_corr.sort_values(by="correlation_all",ascending=False, inplace=True) # sort by correlation
    df_red_corr = df_red_corr[df_red_corr['feature_1'] != df_red_corr['feature_2']] # Remove self correlation
    df_red_corr = df_red_corr.drop_duplicates(subset='correlation_all')

    red_features_corr = list(df_red_corr[abs(df_red_corr['correlation_all'])>=corr_cutoff]['feature_1']) ## abs so we also consider the negative corrs


    df_red = df_red.drop(columns=red_features_corr) ## dropping the highly correlated columns

    cols_removed = [column for column in df.columns if column not in df_red.columns]

    print(f"\n✅ data cleaned, {len(cols_removed)} columns removed")

    return df_red, cols_removed

def get_chunk(source_name: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """

    if "processed" in source_name:
        columns = None
        dtypes = DTYPES_PROCESSED_OPTIMIZED
    else:
        columns = COLUMN_NAMES_RAW
        if os.environ.get("DATA_SOURCE") == "big query":
            dtypes = DTYPES_RAW_OPTIMIZED
        else:
            dtypes = DTYPES_RAW_OPTIMIZED_HEADLESS

    if os.environ.get("DATA_SOURCE") == "big query":

        chunk_df = get_bq_chunk(table=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                verbose=verbose)

        return chunk_df

    chunk_df = get_pandas_chunk(path=source_name,
                                index=index,
                                chunk_size=chunk_size,
                                dtypes=dtypes,
                                columns=columns,
                                verbose=verbose)

    return chunk_df


def save_chunk(destination_name: str,
               is_first: bool,
               data: pd.DataFrame) -> None:
    """
    save chunk
    """

    if os.environ.get("DATA_SOURCE") == "big query":

        save_bq_chunk(table=destination_name,
                      data=data,
                      is_first=is_first)

        return

    save_local_chunk(path=destination_name,
                     data=data,
                     is_first=is_first)
