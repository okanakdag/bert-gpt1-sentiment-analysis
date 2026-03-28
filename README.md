# Sentiment Classification Fine-tuning with BERT and GPT-1

This project compares two pretrained transformer models, BERT and GPT-1, on English sentiment classification using the IMDb movie review dataset.

The implementation covers:
- data loading and preprocessing
- tokenizer-based dataset preparation
- fine-tuning for binary classification
- evaluation with accuracy, precision, recall, and F1
- comparison across multiple training-data settings

In the saved experiments, BERT performs better overall on the IMDb sentiment task, especially on the full-data setting, 
while GPT-1 remains competitive and occasionally gets very close in lower-data comparisons. The full write-up is available in `report.md`.

## Highlights

- BERT and GPT-1 were fine-tuned on the same dataset and evaluated with the same metrics.
- Saved runs include full-data and lower-data settings.
- Results are tracked with descriptive filenames that include training fraction, learning rate, and seed.
- A cross-run summary is available in `results/comparison.md`.

## Project Structure

```text 
.
|-- README.md
|-- report.md
|-- requirements.txt
|-- data/
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocess.py
|   |-- dataset.py
|   |-- train_bert.py
|   |-- train_gpt.py
|   |-- evaluate.py
|   `-- utils.py
`-- results/
    |-- comparison.md
    |-- bert/
    |   `-- models/
    `-- gpt/
        `-- models/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the dataset:
[IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

3. Place the CSV file here:

```text
data/IMDB Dataset.csv
```

## Run

Train BERT:

```bash
python src/train_bert.py
```

Train GPT-1:

```bash
python src/train_gpt.py
```

The current configuration is controlled from `src/config.py`.

## Results

- `results/bert/`: saved BERT metric files
- `results/bert/models/`: saved BERT checkpoints
- `results/gpt/`: saved GPT-1 metric files
- `results/gpt/models/`: saved GPT-1 checkpoints
- `results/comparison.md`: summary comparison across saved runs

Filename format example:

- `bert_train100_lr2e5_seed42.txt`
- `gpt_train10_lr2e5_seed42.txt`

Meaning:

- `bert` / `gpt`: model name
- `train100`, `train10`, `train5`, `train1`: percentage of training data used
- `lr2e5`: learning rate `2e-5`
- `seed42`: random seed `42`

Repeated runs automatically get suffixes such as `_2` to avoid overwriting previous files.

## References

- [BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [GPT-1 documentation](https://huggingface.co/docs/transformers/model_doc/openai-gpt)
