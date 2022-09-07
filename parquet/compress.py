from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from parquet.bigquery_chunks import get_bq_chunk, numeric_conversion, save_bq_chunk
from colorama import Fore, Style

def compress():

    # iterate on the dataset, by chunks
    chunk_size = 100000
    chunk_id = 0
    row_count = 0
    source_name = "train_data"
    destination_name = f"{source_name}_processed"

    while (True):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_bq_chunk(source_name,
                                  index=chunk_id * chunk_size,
                                  chunk_size=chunk_size)

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_compressed = numeric_conversion(data_chunk)

        # save and append the chunk
        is_first = chunk_id == 0

        save_bq_chunk(destination_name,
                      data=data_chunk_compressed,
                      is_first=is_first)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for processing 👌")
        return None

    print(f"\n✅ data processed saved entirely: {row_count} rows processed")

    return None
