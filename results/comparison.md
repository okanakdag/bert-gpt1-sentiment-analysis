# Model Comparison

Following compared runs use:
- `seed: 42`
- `learning rate: 2e-5`
- `dataset_rows: 50000`
- `validation_samples: 5000`
- `test_samples: 10001`
- `num_epochs: 3`

## Overall Trends

- Both models perform best with full training data
- BERT is better on full data and 10% data
- GPT-1 is slightly ahead at 5%

## Best Test F1 By Setting

| Training Data | Best Model | Best Test F1 | Runner-Up | Runner-Up Test F1 | Gap |
|---|---|---:|---|---:|---:|
| Full | BERT | 0.9293 | GPT-1 | 0.9157 | 0.0136 |
| 10% | BERT | 0.9036 | GPT-1 | 0.8964 | 0.0072 |
| 5% | GPT-1 | 0.8942 | BERT | 0.8912 | 0.0030 |

## Test Results Comparison

| Training Data | Train Samples | Model | Test Loss | Accuracy | Precision | Recall | F1 | Best Epoch |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Full | 34999 | BERT | 0.1862 | 0.9283 | 0.9169 | 0.9420 | 0.9293 | 2 |
| Full | 34999 | GPT-1 | 0.2531 | 0.9177 | 0.9383 | 0.8942 | 0.9157 | 3 |
| 10% | 3499 | BERT | 0.3431 | 0.9027 | 0.8953 | 0.9120 | 0.9036 | 3 |
| 10% | 3499 | GPT-1 | 0.2626 | 0.8946 | 0.8812 | 0.9122 | 0.8964 | 1 |
| 5% | 1749 | BERT | 0.3601 | 0.8861 | 0.8531 | 0.9328 | 0.8912 | 3 |
| 5% | 1749 | GPT-1 | 0.2727 | 0.8916 | 0.8732 | 0.9162 | 0.8942 | 2 |

## Validation Results Comparison

| Training Data | Train Samples | Model | Validation Loss | Accuracy | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Full | 34999 | BERT | 0.1981 | 0.9256 | 0.9105 | 0.9440 | 0.9269 |
| Full | 34999 | GPT-1 | 0.2616 | 0.9208 | 0.9402 | 0.8988 | 0.9190 |
| 10% | 3499 | BERT | 0.3463 | 0.8982 | 0.8890 | 0.9100 | 0.8994 |
| 10% | 3499 | GPT-1 | 0.2715 | 0.8932 | 0.8801 | 0.9104 | 0.8950 |
| 5% | 1749 | BERT | 0.3817 | 0.8762 | 0.8363 | 0.9356 | 0.8831 |
| 5% | 1749 | GPT-1 | 0.2753 | 0.8866 | 0.8654 | 0.9156 | 0.8898 |
