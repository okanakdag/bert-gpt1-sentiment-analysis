
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import OpenAIGPTForSequenceClassification, OpenAIGPTTokenizer

from config import (
    BATCH_SIZE,
    GPT_METRICS_PATH,
    GPT_MODEL_NAME,
    GPT_MODEL_PATH,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_EPOCHS,
    NUM_LABELS,
    RANDOM_SEED,
    TRAIN_FRACTION,
)
from data_loader import inspect_dataset, load_raw_data, map_labels_if_needed
from dataset import SentimentDataset
from evaluate import compute_classification_metrics, save_metrics
from preprocess import prepare_features_and_labels, sample_training_data, split_data
from utils import get_device, get_unique_output_paths, load_model, save_model, set_seed


def create_gpt_components():
    """
    GPT-1 does not include a padding token by default, so we add one.
    That makes it easier to batch many examples together.
    """
    tokenizer = OpenAIGPTTokenizer.from_pretrained(GPT_MODEL_NAME)
    model = OpenAIGPTForSequenceClassification.from_pretrained(
        GPT_MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

# Adapt review format to enable classification for an autoregressive model 
def format_prompt(text):
    cleaned_text = str(text).strip().replace("\n", " ")
    return f"Review: {cleaned_text}\nSentiment:"


def create_dataloader(texts, labels, tokenizer, batch_size, shuffle):
    """
    Build a DataLoader for one GPT dataset split.

    Before tokenization, each text is wrapped in the prompt format above.
    """
    prompted_texts = [format_prompt(text) for text in texts]
    dataset = SentimentDataset(
        texts=prompted_texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, optimizer, device):

    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


def validate(model, dataloader, device):

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


def build_metrics_report(dataset_summary, split_sizes, best_epoch, best_validation_metrics, test_metrics):

    return {
        "dataset_rows": dataset_summary["num_rows"],
        "num_epochs": NUM_EPOCHS,
        "train_fraction": TRAIN_FRACTION,
        "train_samples": split_sizes["train"],
        "validation_samples": split_sizes["validation"],
        "test_samples": split_sizes["test"],
        "best_epoch": best_epoch,
        "validation_loss": best_validation_metrics["loss"],
        "validation_accuracy": best_validation_metrics["accuracy"],
        "validation_precision": best_validation_metrics["precision"],
        "validation_recall": best_validation_metrics["recall"],
        "validation_f1": best_validation_metrics["f1"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    }


def main():

    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Learning rate: {LEARNING_RATE}")

    metrics_path, model_path = get_unique_output_paths(GPT_METRICS_PATH, GPT_MODEL_PATH)

    raw_dataframe = load_raw_data()
    dataset_summary = inspect_dataset(raw_dataframe)
    numeric_dataframe = map_labels_if_needed(raw_dataframe)
    split_datasets = split_data(numeric_dataframe)
    split_datasets["train"] = sample_training_data(split_datasets["train"])

    print(f"Using {TRAIN_FRACTION * 100:.0f}% of the training split.")

    train_texts, train_labels = prepare_features_and_labels(split_datasets["train"])
    validation_texts, validation_labels = prepare_features_and_labels(split_datasets["validation"])
    test_texts, test_labels = prepare_features_and_labels(split_datasets["test"])

    tokenizer, model = create_gpt_components()
    model.to(device)

    train_dataloader = create_dataloader(
        train_texts,
        train_labels,
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_dataloader = create_dataloader(
        validation_texts,
        validation_labels,
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_dataloader = create_dataloader(
        test_texts,
        test_labels,
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_epoch = 0
    best_validation_metrics = None
    best_validation_f1 = -1.0

    for epoch_index in range(NUM_EPOCHS):
        epoch_number = epoch_index + 1
        print(f"\nEpoch {epoch_number}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        validation_metrics = validate(model, validation_dataloader, device)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_metrics['loss']:.4f}")
        print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Validation precision: {validation_metrics['precision']:.4f}")
        print(f"Validation recall: {validation_metrics['recall']:.4f}")
        print(f"Validation F1: {validation_metrics['f1']:.4f}")

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_validation_metrics = validation_metrics
            best_epoch = epoch_number
            save_model(model, model_path)

    if best_validation_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")

    model = load_model(model, model_path, device)
    model.to(device)

    test_metrics = validate(model, test_dataloader, device)

    print("\nBest validation epoch:", best_epoch)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

    metrics_report = build_metrics_report(
        dataset_summary=dataset_summary,
        split_sizes={
            "train": len(split_datasets["train"]),
            "validation": len(split_datasets["validation"]),
            "test": len(split_datasets["test"]),
        },
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        test_metrics=test_metrics,
    )
    save_metrics(metrics_report, metrics_path)
    print(f"GPT metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
