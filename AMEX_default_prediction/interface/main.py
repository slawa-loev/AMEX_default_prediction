from AMEX_default_prediction.ml_logic.data import (clean_data,
                                    get_chunk,
                                    save_chunk)

from ml_logic.params import (CHUNK_SIZE,
                                      DATASET_SIZE,
                                      VALIDATION_DATASET_SIZE)

from ml_logic.preprocessor import preprocess_features

import numpy as np
import pandas as pd

from colorama import Fore, Style


def preprocess(source_type='train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    print("\n⭐️ use case: preprocess")

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"

    while (True):

        print(Fore.BLUE + f"\nProcessing chunk n{chunk_id}..." + Style.RESET_ALL)

        data_chunk = get_chunk(source_name=source_name,
                               index=chunk_id * CHUNK_SIZE,
                               chunk_size=CHUNK_SIZE)

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print(Fore.BLUE + "\nNo cleaned data in latest chunk..." + Style.RESET_ALL)
            break

        X_chunk = data_chunk_cleaned.drop("default", axis=1)
        y_chunk = data_chunk_cleaned[["default"]]

        X_processed_chunk = preprocess_features(X_chunk)

        data_processed_chunk = pd.DataFrame(
        np.concatenate((X_processed_chunk, y_chunk), axis=1))

        # save and append the chunk
        is_first = chunk_id == 0

        save_chunk(destination_name=destination_name,
                   is_first=is_first,
                   data=data_processed_chunk)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the preprocessing 👌")
        return None

    print(f"\n✅ data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None

def train():
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ use case: train")

    from AMEX_default_prediction.ml_logic.model import (initialize_model, compile_model, train_model)
    from AMEX_default_prediction.ml_logic.registry import load_model, save_model
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # load a validation set common to all chunks, used to early stop model training
    data_val_processed = get_chunk(
        source_name=f"val_processed_{VALIDATION_DATASET_SIZE}",
        index=0,  # retrieve from first row
        chunk_size=None)  # retrieve all further data

    if data_val_processed is None:
        print("\n✅ no data to train")
        return None

    data_val_processed = data_val_processed.to_numpy()

    X_val_processed = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]

    model = None
    # model params
    # todo

    # iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_val_list = []

    while (True):

        print(Fore.BLUE + f"\nLoading and training on preprocessed chunk n°{chunk_id}..." + Style.RESET_ALL)

        data_processed_chunk = get_chunk(source_name=f"train_processed_{DATASET_SIZE}",
                                         index=chunk_id * CHUNK_SIZE,
                                         chunk_size=CHUNK_SIZE)

        # check whether data source contain more data
        if data_processed_chunk is None:
            print(Fore.BLUE + "\nNo more chunk data..." + Style.RESET_ALL)
            break

        data_processed_chunk = data_processed_chunk.to_numpy()

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]

        # increment trained row count
        chunk_row_count = data_processed_chunk.shape[0]
        row_count += chunk_row_count

        # initialize model
        if model is None:
            model = initialize_model(X_train_chunk)

        # train the model incrementally

        #todo
        model, history = train_model(model,
                                     X_train_chunk,
                                     y_train_chunk
                                   )

        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(f"chunk MAE: {round(metrics_val_chunk,2)}")

        # check if chunk was full
        if chunk_row_count < CHUNK_SIZE:
            print(Fore.BLUE + "\nNo more chunks..." + Style.RESET_ALL)
            break

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    # return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    print(f"\n✅ trained on {row_count} rows with MAE: {round(val_mae, 2)}")

    params = dict(
        # model parameters
        # todo

        # package behavior
        context="train",
        chunk_size=CHUNK_SIZE,
        # data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=row_count,    )

    # save model
    save_model(model=model, params=params, metrics=dict(mae=val_mae))

    return val_mae


def evaluate():
    """
    Evaluate the performance of the latest production model on new data
    """

    print("\n⭐️ use case: evaluate")

    from AMEX_default_prediction.ml_logic.model import evaluate_model
    from AMEX_default_prediction.ml_logic.registry import load_model, save_model
    # load new data
    new_data = get_chunk(source_name=f"val_processed_{DATASET_SIZE}",
                         index=0,
                         chunk_size=None)  # retrieve all further data

    if new_data is None:
        print("\n✅ no data to evaluate")
        return None

    new_data = new_data.to_numpy()

    X_new = new_data[:, :-1]
    y_new = new_data[:, -1]

    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    # save evaluation
    params = dict(        # package behavior
        context="evaluate",
        # data source
        training_set_size=DATASET_SIZE,
        val_set_size=VALIDATION_DATASET_SIZE,
        row_count=len(X_new))

    save_model(params=params, metrics=dict(mae=mae))

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ use case: predict")

    from AMEX_default_prediction.ml_logic.registry import load_model

    if X_pred is None:

        print('please provde dataset')

    model = load_model()

    X_processed = preprocess_features(X_pred)

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    preprocess()
    preprocess(source_type='val')
    train()
    pred()
    evaluate()
