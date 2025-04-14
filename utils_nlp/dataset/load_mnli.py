# utils_nlp/dataset/load_mnli.py
from datasets import load_dataset

def load_mnli_splits(cache_dir="data_cache"):
    """
    Downloads (once) and returns Multi‑NLI train/validation as Pandas DataFrames.
    The Hugging Face 'datasets' library handles caching and retries.
    """
    ds = load_dataset("multi_nli", cache_dir=cache_dir)
    train_df = ds["train"].to_pandas()
    dev_df   = ds["validation_matched"].to_pandas()
    return train_df, dev_df
