# utils_nlp/dataset/load_mnli.py
from datasets import load_dataset

def load_mnli_splits(cache_dir="data_cache"):
    ds = load_dataset("multi_nli", cache_dir=cache_dir)
    train_df = ds["train"].to_pandas()
    dev_df   = ds["validation_matched"].to_pandas()

    # harmonise column names + label values
    rename = {"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label"}
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    train_df = train_df.rename(columns=rename)
    dev_df   = dev_df.rename(columns=rename)
    train_df["gold_label"] = train_df["gold_label"].map(label_map)
    dev_df["gold_label"]   = dev_df["gold_label"].map(label_map)
    return train_df, dev_df
