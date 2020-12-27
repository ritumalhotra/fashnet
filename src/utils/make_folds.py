import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import click


@click.command()
@click.option("--filepath", prompt="Filepath to data")
def main(filepath):
    df = pd.read_csv(filepath)
    df.loc[:, "kfold"] = 1

    df = df.sample(frac=1).reset_index(drop=True)

    X = df["img_path"].values
    y = df.iloc[:, 1:].values

    kfold = MultilabelStratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kfold.split(X, y)):
        print("Train: ", trn_, "Val: ", val_)
        df.loc[val_, "kfold"] = fold

    print(df["kfold"].value_counts())
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
