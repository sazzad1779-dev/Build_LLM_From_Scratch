import sentencepiece as spm
from pathlib import Path
import unicodedata
import re
import pandas as pd

# ---------------------------
# Text normalization
# ---------------------------
def normalize_text(text: str) -> str:
    """Normalize text for tokenizer evaluation."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u0000-\u001F\u007F]", "", text)  # Remove control chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Load corpus from multiple formats
# ---------------------------
def load_corpus(file_path: str, file_type: str = "txt", text_column: str = None):
    """
    Load corpus lines from a file for evaluation.
    
    Args:
        file_path: Path to the corpus file.
        file_type: 'txt', 'csv', or 'md'.
        text_column: For CSV, column containing text.
    
    Returns:
        List of normalized text lines.
    """
    lines = []
    file_path = Path(file_path)

    try:
        if file_type in ["txt", "md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = normalize_text(line)
                    if line:
                        lines.append(line)

        elif file_type == "csv":
            df = pd.read_csv(file_path, usecols=[text_column] if text_column else None)
            for text in df[text_column] if text_column else df.iloc[:,0]:
                text = normalize_text(str(text))
                if text:
                    lines.append(text)
    except Exception as e:
        print(f"⚠ Skipping {file_path} due to error: {e}")

    return lines

# ---------------------------
# Tokenizer evaluation
# ---------------------------
def evaluate_tokenizer(lines, sp):
    """
    Evaluate SentencePiece tokenizer metrics: Fertility, CPT, WFR.
    
    Args:
        lines: List of normalized text lines.
        sp: SentencePieceProcessor object.
    
    Returns:
        fertility, cpt, wfr
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    split_words = 0

    for line in lines:
        words = line.split()
        total_words += len(words)
        total_chars += len(line.replace(" ", ""))

        for word in words:
            pieces = sp.encode(word, out_type=str)
            total_tokens += len(pieces)
            if len(pieces) > 1:
                split_words += 1

    fertility = total_tokens / total_words if total_words else 0
    cpt = total_chars / total_tokens if total_tokens else 0
    wfr = split_words / total_words if total_words else 0

    return fertility, cpt, wfr

# ---------------------------
# Interpret results
# ---------------------------
def interpret_results(fertility, cpt, wfr, print_summary=True):
    """
    Print an interpretation report for tokenizer metrics.
    """
    if print_summary:
        print("\n🔎 TOKENIZER EVALUATION REPORT\n")

        print(f"Fertility: {fertility:.3f}")
        if fertility < 1.2:
            print("  → Tokens too large. Risk of over-merging. Consider reducing vocab size.")
        elif fertility <= 2.0:
            print("  → Healthy token-to-word balance. ✅")
        elif fertility <= 2.5:
            print("  → Moderate fragmentation. Slightly increase vocab size.")
        else:
            print("  → Too many tokens per word. Increase vocab size or use unigram model. ❌")

        print(f"\nCPT (Characters Per Token): {cpt:.3f}")
        if cpt < 2:
            print("  → Character-like tokenization. Increase vocab size. ❌")
        elif cpt < 3.5:
            print("  → Slightly fragmented tokens. Acceptable but can improve.")
        elif cpt <= 6:
            print("  → Information-dense tokens. Ideal range. ✅")
        else:
            print("  → Tokens may be too large. Risk of memorization. Consider reducing vocab.")

        print(f"\nWFR (Word Fragmentation Rate): {wfr:.3f}")
        if wfr < 0.2:
            print("  → Very few words split. May be over-merged.")
        elif wfr <= 0.4:
            print("  → Good subword behavior. Balanced splitting. ✅")
        elif wfr <= 0.6:
            print("  → Many words split. Increase vocab size.")
        else:
            print("  → Over-fragmentation. Use larger vocab or unigram model. ❌")

        print("\n📌 Summary Insight:")
        if fertility <= 2.0 and 3.5 <= cpt <= 6 and wfr <= 0.4:
            print("  🎯 Your tokenizer is well-balanced and efficient!")
        else:
            print("  ⚠ Tokenizer needs tuning based on the issues above.")

# ---------------------------
# Complete evaluation pipeline
# ---------------------------
def run_evaluation(model_path, corpus_path, file_type="txt", text_column=None):
    """
    Run full evaluation on a tokenizer and corpus.
    
    Returns: fertility, cpt, wfr
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    lines = load_corpus(corpus_path, file_type=file_type, text_column=text_column)
    fertility, cpt, wfr = evaluate_tokenizer(lines, sp)
    interpret_results(fertility, cpt, wfr)
    
    return fertility, cpt, wfr

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    model_path = "tokenizer_models/my_tokenizer.model"
    corpus_path = "data/corpus.txt"

    run_evaluation(model_path, corpus_path)
