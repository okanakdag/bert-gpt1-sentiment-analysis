# Convert raw data to tensor based format as a class

import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    # Return one sample as a dictionary of tensors
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        # `return_tensors="pt"` asks the tokenizer to give PyTorch tensors back.
        # Because we tokenize one sample at a time here, the tensors have an
        # extra batch dimension of size 1, so we remove it with `squeeze(0)`.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
