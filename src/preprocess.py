# Prepare raw text data before tokenization.


import re

from sklearn.model_selection import train_test_split

from config import LABEL_COLUMN, RANDOM_SEED, TEST_SIZE, TEXT_COLUMN, TRAIN_FRACTION, VALIDATION_SIZE


def clean_text(text):
    cleaned_text = str(text)
    cleaned_text = cleaned_text.replace("<br />", " ")
    cleaned_text = cleaned_text.replace("<br/><br/>", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def split_data(dataframe, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN):
    working_dataframe = dataframe.copy()
    working_dataframe[text_column] = working_dataframe[text_column].apply(clean_text)

    # First split: keep the training data separate from the temporary holdout set.
    train_df, temp_df = train_test_split(
        working_dataframe,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=working_dataframe[label_column],
    )

    # Second split: divide the temporary holdout set into validation and test.
    validation_fraction_of_holdout = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 - validation_fraction_of_holdout,
        random_state=RANDOM_SEED,
        stratify=temp_df[label_column],
    )

    # Resetting indices keeps later dataset access simple and predictable.
    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def sample_training_data(dataframe, train_fraction=TRAIN_FRACTION, label_column=LABEL_COLUMN):
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be greater than 0 and less than or equal to 1.")

    if train_fraction >= 1.0:
        return dataframe.reset_index(drop=True)

    sampled_dataframe, _ = train_test_split(
        dataframe,
        train_size=train_fraction,
        random_state=RANDOM_SEED,
        stratify=dataframe[label_column],
    )
    return sampled_dataframe.reset_index(drop=True)


def prepare_features_and_labels(dataframe, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN):
    texts = dataframe[text_column].tolist()
    labels = dataframe[label_column].tolist()
    return texts, labels
