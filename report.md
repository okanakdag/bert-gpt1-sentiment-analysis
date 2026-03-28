## Part 1 - Dataset Selection

The IMDb Large Movie Review Dataset was used for this sentiment analysis task.

- Download source I used: [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
- Original paper: Maas et al., [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) (ACL 2011)
- Number of samples: **50,000**

The dataset consists of English-language movie reviews labeled for sentiment classification.

### Label Classes

Raw data labels are **positive** and **negative**. In the implementation, these labels are mapped to numeric classes:
- `positive -> 1`
- `negative -> 0`

### Example Samples

Raw format: `"review","sentiment"`

"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked...",positive

"A wonderful little production. The filming technique is very unassuming...",positive

### Train / Validation / Test Split

The entire dataset was used initially, and a custom split was created during preprocessing:

- **Train:** 34,999 samples
- **Validation:** 5,000 samples
- **Test:** 10,001 samples

This corresponds approximately to:
- **70% training**
- **10% validation**
- **20% test**

A configuration option was later added to allow training on a subset of the data for faster experimentation.
 This is reflected in the naming of result files.

## Part 2 - Fine-Tuning BERT

### Data Preprocessing and Tokenization

The preprocessing steps are:

1. Load the CSV file with `pandas`
2. Map sentiment labels from text to numbers
   - `positive -> 1`
   - `negative -> 0`
3. Apply light text cleaning
   - replace HTML break tags such as `<br />`
   - remove repeated extra spaces
4. Split the dataset into train, validation, and test sets
5. Sample the configured training fraction when needed
6. Tokenize the review text with the BERT tokenizer

Tokenization is done with `BertTokenizer` from Hugging Face.
Each review is converted into:

- `input_ids`
- `attention_mask`
- `label`

The implementation uses padding and truncation to a fixed maximum sequence length.

### Model Architecture Used for Classification

For BERT, I used:

- `BertTokenizer`
- `BertForSequenceClassification`

The pretrained base model is `bert-base-uncased`.

`BertForSequenceClassification` extends the pretrained BERT encoder with a classification head on top.
 This version of the model outputs class scores for the sentiment classes. In this project, the two classes are:

- `0 = negative`
- `1 = positive`

During prediction, the class with the higher score is selected.

### Training Configuration

- Optimizer: `AdamW`
- Learning rate: `2e-5`
- Batch size: `16`
- Number of epochs: `3`

Other important settings:

- Maximum sequence length: `256`
- Random seed: `42`

### Evaluation Metrics

The following evaluation metrics are computed:

- Accuracy
- Precision
- Recall
- F1 score

These metrics are calculated on the validation and test sets using the same shared evaluation code for fair comparison.

### Reported Results

For the full-data BERT run (`bert_train100_lr2e5_seed42.txt`), the test results are:

- Accuracy: **0.9283**
- Precision: **0.9169**
- Recall: **0.9420**
- F1 score: **0.9293**

### How BERT Is Adapted for Classification Tasks

BERT is originally a pretrained language model, but for this homework it is adapted for sentiment 
classification by using [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification).

This adds a classification head on top of the pretrained BERT encoder. The review text is first 
tokenized into BERT input format, then passed through the transformer. The model uses the pooled 
representation of the input sequence to produce two output scores, one for negative sentiment and 
one for positive sentiment. The predicted label is the class with the higher score.

So, instead of being used only for pretraining objectives such as masked language modeling, 
BERT is fine-tuned here as a supervised classifier for binary sentiment prediction.

## Part 3 - Fine-Tuning GPT-1

### How GPT-1 Is Adapted for Sentiment Classification

GPT-1 is originally a pretrained autoregressive language model, which means it is
designed to predict the next token in a sequence. For this homework, it is adapted
for classification by using [`OpenAIGPTForSequenceClassification`](https://huggingface.co/docs/transformers/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification).

This keeps GPT-1 as the base transformer model, but adds a sequence-classification
head on top. In Hugging Face's implementation, the classification decision is based
on the final token representation of the input sequence. The model then produces
class scores for the two sentiment labels:

- `0 = negative`
- `1 = positive`

The predicted class is the one with the larger score.

### How Input Prompts Are Structured

For the GPT-1 experiment, each review is converted into a prompt-style input before
tokenization. The prompt format used in the implementation is:

`Review: <review text>`
`Sentiment:`

For example:

`Review: This movie was entertaining and well acted.`
`Sentiment:`

This input format was used to make the sentiment classification task more explicit
for GPT-1 before tokenization.

### Training Objective Used During Fine-Tuning

During fine-tuning, GPT-1 is trained with a supervised sequence-classification
objective. The model receives tokenized inputs and the true sentiment label for each
example. The sequence-classification head then learns to assign a higher score to
the correct sentiment class.

So, although GPT-1 is pretrained with autoregressive language modeling, the
fine-tuning stage in this homework uses classification loss for binary sentiment
prediction.

### Evaluation Metrics

The same evaluation metrics used for BERT are also used for GPT-1:

- Accuracy
- Precision
- Recall
- F1 score

### Reported Results

For the full-data GPT-1 run (`gpt_train100_lr2e5_seed42.txt`), the test results are:

- Accuracy: **0.9177**
- Precision: **0.9383**
- Recall: **0.8942**
- F1 score: **0.9157**

## Part 4 - Model Comparison

### Differences in Architecture

BERT and GPT-1 are both transformer-based pretrained language models, but they use different architectures.

- **BERT** is bidirectional. It uses context from both the left and the right at the same time.
- **GPT-1** is autoregressive. It processes text from left to right and predicts the next token 
using only previous tokens.

Because of this, BERT is more naturally suited for classification tasks that require understanding 
the full input sequence. GPT-1 can be adapted for classification, but its architecture is originally 
designed for text generation.

### Differences in Training Objectives

Their pretraining objectives are also different:

- **BERT** is pretrained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
- **GPT-1** is pretrained with autoregressive next-token prediction.

During fine-tuning in this homework, both models are trained with a classification objective, but their 
pretraining objectives still affect how well they transfer to sentiment analysis.

### Performance Differences on the Dataset

On the full dataset, the results are:

- **BERT test F1:** `0.9293`
- **GPT-1 test F1:** `0.9157`

BERT performs better than GPT-1 on the full IMDb sentiment classification task.

In the lower-data runs:

- At **10%**, BERT is still slightly better.
- At **5%**, GPT-1 is very slightly better. but the gap is small and may reflect sampling or 
training variability rather than a consistent advantage.

Overall, the difference is small, but BERT performs better in most settings.

### Test Results Comparison Table

| Training Data | Train Samples | Model | Test Loss | Accuracy | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Full | 34999 | BERT | 0.1862 | 0.9283 | 0.9169 | 0.9420 | 0.9293 |
| Full | 34999 | GPT-1 | 0.2531 | 0.9177 | 0.9383 | 0.8942 | 0.9157 |
| 10% | 3499 | BERT | 0.3431 | 0.9027 | 0.8953 | 0.9120 | 0.9036 |
| 10% | 3499 | GPT-1 | 0.2626 | 0.8946 | 0.8812 | 0.9122 | 0.8964 |
| 5% | 1749 | BERT | 0.3601 | 0.8861 | 0.8531 | 0.9328 | 0.8912 |
| 5% | 1749 | GPT-1 | 0.2727 | 0.8916 | 0.8732 | 0.9162 | 0.8942 |

### Advantages and Disadvantages of Each Model

**BERT**

Advantages:
- Well suited for sentence and sequence classification
- Uses bidirectional context, which improves full-sequence understanding for tasks like 
sentiment classification
- Achieves slightly better results in this project

Disadvantages:
- Less flexible than generative models for open-ended tasks
- Requires task-specific fine-tuning for classification

**GPT-1**

Advantages:
- Can be adapted to many tasks
- Performs competitively on this dataset despite being less naturally suited for classification

Disadvantages:
- Originally designed for generation rather than classification
- Uses one-directional context
- Performed slightly worse in this project

### Why One Model Performs Better for This Task

BERT performs better overall because sentiment classification benefits from access to full bidirectional context. 
Movie reviews often contain long dependencies, negation, and sentiment shifts, and BERT is better suited to capture 
these patterns for classification.

GPT-1 remains competitive, but it is originally optimized for autoregressive prediction rather than sentence-level 
classification. For that reason, it performs slightly worse than BERT on this task.

## Part 5 - Conceptual Questions

### Question 1 - Multi-Head Attention

Multi-Head Attention is a core mechanism in the transformer architecture. It allows each token to attend to other 
tokens in the sequence when building its contextual representation.

#### Query, Key, and Value Matrices

For each token representation, the model creates three vectors through learned linear projections:

- **Query (Q):** what the token is looking for
- **Key (K):** what each token offers
- **Value (V):** the information carried by each token

#### Attention Formula

The scaled dot-product attention formula is:

`Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V`

This works as follows:

- compute similarity scores between queries and keys
- scale by `sqrt(d_k)` to keep values stable
- apply `softmax` to obtain attention weights
- use those weights to combine the value vectors

#### Role of Multiple Attention Heads

Instead of computing attention once, the transformer computes it multiple times in parallel. Each head has its own 
learned projections, so different heads can capture different relationships.

For example, one head may focus on local context, while another captures long-range dependencies or syntactic patterns.

#### Why Multi-Head Attention Improves Representation Learning

Multi-head attention improves representation learning because it allows the model to capture multiple types of 
relationships at the same time. This produces richer contextual representations than a single attention computation.

### Question 2 - Loss Function in Transformer Machine Translation

In machine translation, the training objective is to predict the correct next target token at each decoding step.

#### Training Objective

Given a source sentence and the previous correct target tokens, the model learns to maximize the probability of 
the next correct target token.

#### Cross-Entropy Loss Function

The standard loss function is cross-entropy loss. At each target position, the model produces a probability 
distribution over the vocabulary. Cross-entropy compares this predicted distribution with the true target token.

Low cross-entropy means the model assigns high probability to the correct token. High cross-entropy means the 
model assigns more probability to incorrect tokens.

The total training loss is typically the sum or average of the cross-entropy losses over all target positions 
in the sequence.

#### Which Parameters Are Updated During Training

During backpropagation, all trainable model parameters are updated, including:

- encoder parameters
- decoder parameters
- attention projection matrices
- feed-forward layer parameters
- embedding matrices
- output projection parameters

The transformer is therefore trained end-to-end.

### Question 3 - Why Masked Self-Attention Is Used in the Decoder

Masked self-attention is used in the decoder to preserve the autoregressive property of sequence generation.

#### The Autoregressive Property

In autoregressive generation, token `t` should be predicted using only tokens before it, not future tokens.

#### How Masking Ensures Correct Training Behavior

The decoder applies a causal mask so that each position can attend only to itself and earlier positions. 
Future tokens are blocked.

Without masking, the model could look at future target tokens during training, which would make training 
inconsistent with actual generation. Masking prevents this and ensures that training matches inference.

### Question 4 - BERT Pretraining Tasks

BERT uses two main pretraining tasks:

#### Masked Language Modeling (MLM)

Some input tokens are masked, and the model is trained to predict the original masked tokens using the surrounding context.

This allows BERT to learn bidirectional representations, because the prediction can use both left and right context.

#### Next Sentence Prediction (NSP)

BERT is also trained to predict whether one sentence logically follows another. This helps it learn relationships between sentences.

#### Why Standard Language Modeling Is Not Used for BERT

Standard left-to-right language modeling only allows access to previous tokens. BERT is designed to be bidirectional, 
so that objective would restrict its ability to use full context.

Masked Language Modeling solves this by hiding selected tokens and training the model to recover them from both directions.

### Question 5 - GPT-1 Pretraining Task

GPT-1 is pretrained with autoregressive language modeling.

#### Autoregressive Language Modeling

The model reads text from left to right and learns to predict the next token in the sequence.

#### How GPT-1 Predicts the Next Token

At position `t`, GPT-1 uses the previous tokens to compute a probability distribution over the vocabulary for 
the next token. Training increases the probability assigned to the correct next token.

This process is repeated across the sequence, allowing GPT-1 to learn general language patterns from unlabeled text.

#### How Different Downstream Tasks Are Handled During Fine-Tuning

During fine-tuning, GPT-1 is adapted to downstream tasks by modifying the input format and training it on labeled examples.

For classification tasks such as sentiment analysis:

- the input can be formatted to make the task explicit
- a task-specific classification head can be added
- the model is fine-tuned with supervised loss

In this way, GPT-1 keeps its pretrained language knowledge and adapts it to the downstream task.