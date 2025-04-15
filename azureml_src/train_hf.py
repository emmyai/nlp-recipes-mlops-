# train_hf.py
#
# Fine‑tunes a BERT‑style model on Multi‑NLI and writes:
#   • outputs/results.json            – simple metrics dict
#   • outputs/model/                  – Hugging‑Face model + config + tokenizer

import argparse, json, os, pathlib
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_mnli(train_files=None, test_files=None, cache_dir="data_cache"):
    """
    Returns train/validation DataFrames.
    If AzureML passed --train_files/--test_files, load those CSV/Parquet files;
    otherwise pull the corpus from the Hugging‑Face hub.
    """
    if train_files and test_files:
        train_df = pd.concat([pd.read_csv(f) for f in train_files])
        val_df   = pd.concat([pd.read_csv(f) for f in test_files])
    else:
        ds = load_dataset("multi_nli", cache_dir=cache_dir)
        train_df = ds["train"].to_pandas()
        val_df   = ds["validation_matched"].to_pandas()

    # harmonise columns
    rename = {"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label"}
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    train_df = train_df.rename(columns=rename)
    val_df   = val_df.rename(columns=rename)
    train_df["gold_label"] = train_df["gold_label"].map(label_map)
    val_df["gold_label"]   = val_df["gold_label"].map(label_map)
    return train_df, val_df


def preprocess_df(df, tokenizer, max_len=128):
    enc = tokenizer(
        df["sentence1"].tolist(),
        df["sentence2"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    labels = torch.tensor(
        df["gold_label"].map({"entailment": 0, "neutral": 1, "contradiction": 2}).tolist()
    )
    return enc, labels


class NLNIDataset(torch.utils.data.Dataset):
    def __init__(self, enc, lbl): self.enc, self.lbl = enc, lbl
    def __len__(self): return len(self.lbl)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.lbl[idx]
        return item


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", nargs="*", default=None)
    parser.add_argument("--test_files",  nargs="*", default=None)
    parser.add_argument("--fast_run",    action="store_true")
    parser.add_argument("--epochs",      type=int, default=1)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet warning

    # 1) data ----------------------------------------------------------------
    train_df, val_df = load_mnli(args.train_files, args.test_files)

    if args.fast_run:
        train_df = train_df.sample(10_000, random_state=0)
        val_df   = val_df.sample(2_000,  random_state=0)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_enc, train_lbl = preprocess_df(train_df, tokenizer)
    val_enc,   val_lbl   = preprocess_df(val_df,   tokenizer)

    # 2) model ---------------------------------------------------------------
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    args_tr = TrainingArguments(
        output_dir                   = "outputs/model",
        per_device_train_batch_size  = 16,
        per_device_eval_batch_size   = 16,
        num_train_epochs             = args.epochs,
        evaluation_strategy          = "epoch",
        logging_steps                = 20,
        load_best_model_at_end       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = args_tr,
        train_dataset = NLNIDataset(train_enc, train_lbl),
        eval_dataset  = NLNIDataset(val_enc,   val_lbl),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # 3) save artefacts ------------------------------------------------------
    pathlib.Path("outputs").mkdir(exist_ok=True)
    json.dump(eval_metrics, open("outputs/results.json", "w"), indent=2)

    model.save_pretrained("outputs/model")
    tokenizer.save_pretrained("outputs/model")


if __name__ == "__main__":
    main()
