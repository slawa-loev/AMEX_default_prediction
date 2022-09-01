
from google.cloud import bigquery

import pandas as pd

from colorama import Fore, Style
import os

from AMEX_default_prediction.ml_logic.params import PROJECT, DATASET


def get_bq_chunk(table: str,
                 index: int,
                 chunk_size: int,
                 dtypes: dict = None,
                 verbose=True) -> pd.DataFrame:
    """
    return a chunk of a big query dataset table
    format the output dataframe according to the provided data types
    """
    if verbose:
        print(Fore.MAGENTA + f"Source data from big query {table}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})" + Style.RESET_ALL)

    try:

        table = f"{PROJECT}.{DATASET}.train_10k"

        client = bigquery.Client()
        rows = client.list_rows(table, start_index=index+1, max_results=chunk_size)
        df = rows.to_dataframe().astype(dtypes)

        # read_csv(dtypes=...) will silently fail to convert data types, if column names do no match dictionnary key provided.
        if isinstance(dtypes, dict):
            assert dict(df.dtypes) == dtypes



    except pd.errors.EmptyDataError:

        return None  # end of data

    return df





def save_bq_chunk(table: str,
                  data: pd.DataFrame,
                  is_first: bool):
    """
    save a chunk of the raw dataset to big query
    empty the table beforehands if `is_first` is True
    """

    print(Fore.BLUE + f"\nSave data to big query {table}:" + Style.RESET_ALL)




    data.to_gbq(f'{DATASET}.{table}')
