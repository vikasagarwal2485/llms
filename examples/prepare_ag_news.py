"""Download AG News and write a small 4-class split to datasets/ag_news/{train,val,test}.csv.

Usage: python examples/prepare_ag_news.py [--per-class-train 1000] [--per-class-val 100] [--per-class-test 250]
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv"
CLASSES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def fetch(name):
    response = requests.get(f"{BASE_URL}/{name}", timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), header=None, names=["class", "title", "description"])
    text = (df["title"] + ". " + df["description"]).str.replace("\\", " ", regex=False)
    text = text.str.replace(" #36;", "$", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.DataFrame({"text": text, "label": df["class"].map(CLASSES)})


def sample_per_class(df, n, seed):
    return df.groupby("label", group_keys=False).sample(n, random_state=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class-train", type=int, default=1000)
    parser.add_argument("--per-class-val", type=int, default=100)
    parser.add_argument("--per-class-test", type=int, default=250)
    parser.add_argument("--out", default="datasets/ag_news")
    args = parser.parse_args()

    train_pool, test_pool = fetch("train.csv"), fetch("test.csv")
    train = sample_per_class(train_pool, args.per_class_train, seed=123)
    val = sample_per_class(train_pool.drop(train.index), args.per_class_val, seed=123)
    test = sample_per_class(test_pool, args.per_class_test, seed=123)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.sample(frac=1, random_state=123).to_csv(out / f"{name}.csv", index=False)
        print(f"Wrote {len(df)} rows to {out / f'{name}.csv'}")


if __name__ == "__main__":
    main()
