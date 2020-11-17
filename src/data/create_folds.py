import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


if __name__ == "__main__":
    # TODO(Sayar) Finalize train data
    df = pd.read_csv("train.csv")
    print(df.head())
    df.loc[:, "kfold"] = 1

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_id.values
    # TODO(Sayar) Fix column names for dataset
    y = df[[]]

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (trn, val) in enumerate(mskf.split(X, y)):
        print("Train: ", trn, "Val: ", val)
        df.loc[val, "kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv("train_folds.csv", index=False)
