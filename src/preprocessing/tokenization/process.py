from pathlib import Path
import unicodedata
import re
import pandas as pd

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

