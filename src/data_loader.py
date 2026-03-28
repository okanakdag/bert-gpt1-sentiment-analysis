# Loads dataset and converts text labels to numeric

import pandas as pd
from pandas.api.types import is_numeric_dtype

from config import DATASET_PATH, LABEL_COLUMN, TEXT_COLUMN


def load_raw_data(csv_path=DATASET_PATH):
    dataframe = pd.read_csv(csv_path)
    return dataframe


# Returns dataset information
def inspect_dataset(dataframe):
    dataset_summary = {
        "num_rows": len(dataframe),
        "num_columns": len(dataframe.columns),
        "columns": dataframe.columns.tolist(),
        "missing_values": dataframe.isna().sum().to_dict(),
    }

    if LABEL_COLUMN in dataframe.columns:
        dataset_summary["label_counts"] = dataframe[LABEL_COLUMN].value_counts().to_dict()

    return dataset_summary


# Convert text labels to numeric ones for easier training
def map_labels_if_needed(dataframe):

    mapped_dataframe = dataframe.copy()

    if TEXT_COLUMN not in mapped_dataframe.columns or LABEL_COLUMN not in mapped_dataframe.columns:
        raise ValueError(
            f"Expected columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}' to exist in the dataset."
        )

    label_mapping = {
        "negative": 0,
        "positive": 1,
    }

    label_series = mapped_dataframe[LABEL_COLUMN]

    if not is_numeric_dtype(label_series):
        normalized_labels = label_series.astype(str).str.strip().str.lower()
        mapped_dataframe[LABEL_COLUMN] = normalized_labels.map(label_mapping)

    if mapped_dataframe[LABEL_COLUMN].isna().any():
        raise ValueError("Some labels could not be mapped to numeric values.")

    mapped_dataframe[LABEL_COLUMN] = mapped_dataframe[LABEL_COLUMN].astype(int)
    return mapped_dataframe
