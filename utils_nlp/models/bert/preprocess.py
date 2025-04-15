import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input dataset (CSV/TSV)")
    parser.add_argument("--output_data", type=str, help="Path to save preprocessed output")
    args = parser.parse_args()

    print(f"Reading input data from {args.input_data}")
    df = pd.read_csv(os.path.join(args.input_data, "data.csv"), sep="\t")

    # Basic preprocessing: remove nulls and encode labels
    df = df.dropna(subset=["sentence1", "sentence2", "genre"])
    label_map = {"telephone": 0, "government": 1, "travel": 2, "slate": 3, "fiction": 4}
    df["label"] = df["genre"].map(label_map)

    os.makedirs(args.output_data, exist_ok=True)
    output_path = os.path.join(args.output_data, "preprocessed.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved preprocessed data to {output_path}")

if __name__ == "__main__":
    main()
