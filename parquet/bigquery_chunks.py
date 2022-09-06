from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account

# get chunks of data from bigquery
def get_bq_chunk(table: str,
                 index: int,
                 chunk_size: int) -> pd.DataFrame:

    project_name = 'amex-data'
    dataset_name = 'train_data' # to be updated accordingly

    # update key_path to the json file saved in your machine
    key_path = "/Users/isislim/Documents/LeWagon/project_pitch/amex-data-a1386be8bf58.json"
    credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

    table = f"{project_name}.{dataset_name}.{table}"
    client = bigquery.Client(credentials=credentials)
    rows = client.list_rows(table, start_index=index, max_results=chunk_size)
    df = rows.to_dataframe()

    return df

# compress numeric data types
def numeric_conversion(df):
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)

    return df

# save compressed data into a new table in bigquery
def save_bq_chunk(table: str,
                  data: pd.DataFrame,
                  is_first: bool):

    project_name = 'amex-data'
    dataset_name = 'train_data' # to be updated accordingly

    table = f"{project_name}.{dataset_name}.{table}"

    data.columns = [f"_{column}" if type(column) != str else column for column in data.columns]

    key_path = "/Users/isislim/Documents/LeWagon/project_pitch/amex-data-a1386be8bf58.json"
    credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

    client = bigquery.Client(credentials=credentials)

    write_mode = "WRITE_TRUNCATE" if is_first else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data,table,job_config=job_config)
    result = job.result()
