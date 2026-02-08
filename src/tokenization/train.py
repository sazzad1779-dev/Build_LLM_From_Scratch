import sentencepiece as spm
from pathlib import Path
import unicodedata
import re
import pandas as pd
import argparse
from src.tokenization.evaluate import run_evaluation

# ---------------------------
# Text normalization function
# ---------------------------
def normalize_text(text: str) -> str:
    """Normalize text: NFKC unicode, remove control chars, normalize spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u0000-\u001F\u007F]", "", text)  # Remove control chars
    text = re.sub(r"\s+", " ", text).strip()           # Collapse multiple spaces
    return text

# ---------------------------
# Prepare corpus from multiple formats
# ---------------------------
def prepare_corpus(input_dir: str, output_file: str, file_types=None, text_column=None):
    """
    Reads TXT, CSV, and Markdown files from input_dir, normalizes text, and writes to output_file.

    Args:
        input_dir: Path to folder containing files
        output_file: Path to output corpus
        file_types: List of file extensions to read, e.g., ['txt', 'csv', 'md']
        text_column: For CSV files, specify the column to extract text from
    """
    if file_types is None:
        file_types = ['txt', 'csv', 'md']

    input_path = Path(input_dir)
    all_lines = []

    for ext in file_types:
        for file in input_path.glob(f"*.{ext}"):
            try:
                if ext == "txt" or ext == "md":
                    with open(file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = normalize_text(line)
                            if line:
                                all_lines.append(line)

                elif ext == "csv":
                    df = pd.read_csv(file, usecols=[text_column] if text_column else None)
                    for text in df[text_column] if text_column else df.iloc[:,0]:
                        text = normalize_text(str(text))
                        if text:
                            all_lines.append(text)

            except Exception as e:
                print(f"⚠ Skipping {file} due to error: {e}")

    # Write all normalized lines to output file
    with open(output_file, "w", encoding="utf-8") as out:
        for line in all_lines:
            out.write(line + "\n")

    print(f"✅ Corpus prepared: {len(all_lines)} lines written to {output_file}")



# =============================
# TOKENIZER TRAINING
# =============================
def train_tokenizer(args):
    from pathlib import Path

    output_dir = Path(args.model_save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)  # Create folder if not exists

    model_prefix = output_dir / args.model_prefix
    spm.SentencePieceTrainer.train(
        input=args.corpus,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        normalization_rule_name=args.norm_rule,
        split_by_whitespace=True,
        remove_extra_whitespaces=True,

        input_sentence_size=args.sample_size,
        shuffle_input_sentence=True,
        max_sentence_length=args.max_sentence_length,

        byte_fallback=args.byte_fallback,
        hard_vocab_limit=False,

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizer")

    parser.add_argument("--input_dir", type=str, default="data",
                        help="Directory containing text files")
    parser.add_argument("--corpus", type=str, default="corpus.txt",
                        help="Merged training corpus file")
    parser.add_argument("--model_save_dir", type=str, default="tokenizer_models",
                        help="Output model save directory")
    parser.add_argument("--model_prefix", type=str, default="my_tokenizer",
                        help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=16000,
                        help="Vocabulary size")
    parser.add_argument("--model_type", type=str, default="unigram",
                        choices=["unigram", "bpe", "word", "char"],
                        help="SentencePiece model type")
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                        help="Character coverage")
    parser.add_argument("--norm_rule", type=str, default="nmt_nfkc",
                        help="Normalization rule")
    parser.add_argument("--sample_size", type=int, default=10000000,
                        help="Number of sentences sampled for training")
    parser.add_argument("--max_sentence_length", type=int, default=4096,
                        help="Max sentence length")
    parser.add_argument("--byte_fallback", action="store_true",
                        help="Enable byte fallback")

    args = parser.parse_args()

    print("🔹 Preparing corpus...")
    prepare_corpus(args.input_dir, args.corpus)

    print("🔹 Training tokenizer...")
    train_tokenizer(args)

    print("✅ Training complete!")

    print(f"Model saved to: {args.model_save_dir}/{args.model_prefix}.model and .vocab")

    print("Evaluate Fertility, CPT, and WFR on the same corpus to check tokenization quality.")

    model_path = f"{args.model_save_dir}/{args.model_prefix}.model"
    run_evaluation(model_path, args.corpus)
    