# 🧠 Tokenization Metrics Guide

### Understanding Fertility, CPT, WFR, and Normalization (Before Training SentencePiece)

When building a tokenizer for a custom dataset (especially for LLMs), training SentencePiece directly without evaluation can lead to inefficient tokenization.
This guide explains the **core metrics** that help evaluate tokenizer quality **before and after training**.

---

## 📌 Why These Metrics Matter

A tokenizer affects:

* Model context length usage
* Training speed
* Memory efficiency
* Language understanding quality

Bad tokenization = longer sequences + slower models + weaker generalization.

---

# 1️⃣ Fertility

### 🔹 Definition

**Fertility** measures how many tokens are produced per word.

[
\textbf{Fertility} = \frac{N_{tokens}}{N_{words}}
]

### Where:

* (N_{tokens}) = total tokens after tokenization
* (N_{words}) = total words in the corpus

### 🔹 Intuition

How *fragmented* your words become.

### 🔹 Example

Sentence:

```
আমি তোমাকে ভালোবাসি
```

| Tokenization                                  | Tokens | Fertility |
| --------------------------------------------- | ------ | --------- |
| ["আমি","তোমাকে","ভালোবাসি"]                   | 3      | 1.0 ✅     |
| ["আ","মি","তো","মা","কে","ভা","লো","বা","সি"] | 9      | 3.0 ❌     |

### 🔹 Interpretation

| Fertility | Meaning                   |
| --------- | ------------------------- |
| ~1.0      | Word-level                |
| 1.2–2.0   | Good subword tokenization |
| >2.5      | Over-fragmentation        |

---

# 2️⃣ CPT — Characters Per Token

### 🔹 Definition

Average number of characters each token represents.

[
\textbf{CPT} = \frac{N_{char}}{N_{tokens}}
]

### Where:

* (N_{char}) = total characters in corpus
* (N_{tokens}) = total tokens

### 🔹 Intuition

Measures **token information density**.

### 🔹 Example

Text: `বাংলাদেশ`

| Tokenization             | Tokens | CPT   |
| ------------------------ | ------ | ----- |
| ["বাংলাদেশ"]             | 1      | 7.0 ✅ |
| ["বা","ং","লা","দে","শ"] | 5      | 1.4 ❌ |

### 🔹 Interpretation

| CPT       | Meaning                       |
| --------- | ----------------------------- |
| ~1        | Character-level (inefficient) |
| 3.5–6     | Ideal subword tokenizer ✅     |
| Very high | Over-merged tokens            |

---

# 3️⃣ WFR — Word Fragmentation Rate

### 🔹 Definition

Fraction of words split into **two or more tokens**.

[
\textbf{WFR} = \frac{N_{split_words}}{N_{words}}
]

### Where:

* (N_{split_words}) = words producing ≥2 tokens
* (N_{words}) = total words

### 🔹 Example

Sentence:

```
internationalization is hard
```

| Word                 | Tokens | Split? |
| -------------------- | ------ | ------ |
| internationalization | 3      | ✅      |
| is                   | 1      | ❌      |
| hard                 | 1      | ❌      |

[
WFR = \frac{1}{3} = 0.33
]

### 🔹 Interpretation

| WFR     | Meaning                 |
| ------- | ----------------------- |
| 0       | No words split          |
| 0.2–0.4 | Normal subword behavior |
| >0.6    | Too fragmented ❌        |

---

# 4️⃣ Normalization

### 🔹 Definition

A preprocessing transformation applied before tokenization.

[
\boxed{T' = f(T)}
]

Where:

* (T) = original text
* (f) = normalization function
* (T') = normalized text

### 🔹 Purpose

Ensure **text consistency** and reduce vocabulary duplication.

### 🔹 Common Normalization Functions

| Operation                | Effect                       |
| ------------------------ | ---------------------------- |
| Unicode NFC/NFKC         | Standardizes character forms |
| Lowercasing              | A → a                        |
| Whitespace normalization | Multiple spaces → one        |
| Quote normalization      | “ ” → "                      |
| Digit normalization      | ১২৩ → 123 (optional)         |

### 🔹 Example

Raw:

```
আমি   ভালোবাসি!!!
```

Normalized:

```
আমি ভালোবাসি !
```

---

# 🔗 How These Metrics Work Together

| Metric        | Measures                 | Goal          |
| ------------- | ------------------------ | ------------- |
| Fertility     | Tokens per word          | Keep low      |
| CPT           | Info per token           | Moderate–high |
| WFR           | Word splitting frequency | Controlled    |
| Normalization | Text consistency         | Reduce noise  |

---

# 🎯 Ideal Tokenizer Characteristics

✔ Low fertility
✔ CPT between 3.5–6
✔ WFR below 0.4
✔ Proper Unicode normalization
✔ No excessive character-level splitting

---

# 🚀 Why This Matters for LLMs

Better tokenization leads to:

* Shorter input sequences
* Faster training
* Better context utilization
* Lower memory use
* Improved generalization


---

# 🛠 Tokenizer Training with Custom Data

Follow these steps to train a custom SentencePiece tokenizer on your own dataset.

---

## 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo>
```

---

## 2️⃣ Sync package requirements

If you are using **uv**:

```bash
uv sync
```

---

## 3️⃣ Run the training script

```bash
uv run -m src.tokenization.train \
  --input_dir "data/dataset_files" \
  --corpus "data/corpus.txt" \
  --vocab_size 128000 \
  --model_save_dir "tokenizer_models1" \
  --model_type "bpe"
```

> This will:
>
> * Merge and normalize text files from `data/dataset_files`
> * Train a SentencePiece tokenizer with BPE model
> * Save the model and vocab in `tokenizer_models1/`

---

## 4️⃣ Training Options

| Option                                      | Description                                         |
| ------------------------------------------- | --------------------------------------------------- |
| `-h, --help`                                | Show help message                                   |
| `--input_dir INPUT_DIR`                     | Directory containing text files                     |
| `--corpus CORPUS`                           | Path to merged training corpus file                 |
| `--model_save_dir MODEL_SAVE_DIR`           | Directory to save the trained model                 |
| `--model_prefix MODEL_PREFIX`               | Base name for model and vocab files                 |
| `--vocab_size VOCAB_SIZE`                   | Vocabulary size (e.g., 16000, 32000, 128000)        |
| `--model_type {unigram,bpe,word,char}`      | SentencePiece model type                            |
| `--character_coverage CHARACTER_COVERAGE`   | Fraction of characters to include (default: 0.9995) |
| `--norm_rule NORM_RULE`                     | Normalization rule (default: nmt_nfkc)              |
| `--sample_size SAMPLE_SIZE`                 | Number of sentences sampled for training            |
| `--max_sentence_length MAX_SENTENCE_LENGTH` | Max sentence length                                 |
| `--byte_fallback`                           | Enable byte fallback for unseen characters          |

---

✅ **Tip:** Adjust `vocab_size` and `model_type` based on your dataset size and language. After training, you can immediately evaluate the tokenizer using the evaluation script to check metrics like **Fertility, CPT, and WFR**.


