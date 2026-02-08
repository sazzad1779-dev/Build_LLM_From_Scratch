# Build_LLM_From_Scratch

This project demonstrates the **complete process of building a Large Language Model (LLM) from scratch**.

---

## Project Overview

In this repository, we go step by step to:

* Prepare and preprocess datasets
* Train a **custom tokenizer**
* Build Architecture
* Pre-train and evaluation
* Finetune and evaluation
* Deployment setup

---

### 🔹 Prepare and Process Datasets
##### Tokenization Dataset Preparation


##### Pretraining Dataset Preparation

### 🔹 Tokenization

The first part of this project focuses on **tokenization** and creating a **custom tokenizer** for your dataset.

* You will learn how to:

  * Prepare a corpus from TXT, CSV, or Markdown files
  * Train a SentencePiece tokenizer (BPE, Unigram, etc.)
  * Evaluate tokenizer quality using metrics like **Fertility, CPT, and WFR**

* For detailed instructions, see:
  [**Tokenization Guide**](src/tokenization/README.md)

---

### 🔹 Next Steps

After tokenization, the project will cover:

1. Building the LLM architecture
2. Training the model on the prepared dataset
3. Evaluating performance

---

## 🔹 Repository Structure

```
src/
├─ preprocessing/        # Scripts for corpus preparation
├─ tokenization/        # Scripts for tokenization corpus preparation, tokenizer training, and evaluation
├─ modeling/            # Model architecture 
├─ pretraining/            # Model pre-training scripts
├─ finetuning/             # Model finetuning scripts
├─ evaluation/             # Model evaluation scripts
tokenizer_models/        # Trained tokenizer models
README.md               # Root README
```

---

✅ This root README gives an **overview** and links to the detailed tokenization instructions.

