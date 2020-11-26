import os

import pandas as pd
import click


@click.command()
@click.option("--image_path_file", prompt="Image path file")
@click.option("--attribute_file", prompt="Attribute path file")
@click.option("--category_file", prompt="Category path file")
@click.option("--destpath", prompt="Destination path for output")
@click.option("--filename", prompt="Output filename")
def main(image_path_file, attribute_file, category_file, destpath, filename):
    df1 = pd.read_csv(image_path_file, sep=" ", header=None, names=["img_path"])
    df2 = pd.read_csv(attribute_file, sep=" ", header=None)
    df3 = pd.read_csv(category_file, sep=" ")
    result = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
    result["Pattern"] = result.iloc[:, 1:8].idxmax(1)
    result["Pattern"] = result["Pattern"] - 1
    result["Sleeve"] = result.iloc[:, 8:11].idxmax(1)
    result["Sleeve"] = result["Sleeve"] - 8
    result["Length"] = result.iloc[:, 11:14].idxmax(1)
    result["Length"] = result["Length"] - 11
    result["Neckline"] = result.iloc[:, 14:18].idxmax(1)
    result["Neckline"] = result["Neckline"] - 14
    result["Material"] = result.iloc[:, 18:24].idxmax(1)
    result["Material"] = result["Material"] - 18
    result["Fit"] = result.iloc[:, 24:27].idxmax(1)
    result["Fit"] = result["Fit"] - 24
    result.drop(list(range(1, 28)), axis=1, inplace=True)
    result.columns = ["img_path", "Category"] + result.columns.tolist()[2:]

    result.to_csv(os.path.join(destpath, filename), index=False)


if __name__ == "__main__":
    main()
