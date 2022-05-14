import pandas as pd
import click


def read_data(path: str):

    print('Reading data')
    try:
        df = pd.read_csv(path, sep=",", encoding="utf_8")
    except:
        print("Error reading data")
        return

    try:
        df["loan_type"] = df["status"].apply(lambda x: 0 if x in ["A", "C"] else 1)
        df.drop(["status"], axis=1, inplace=True)
    except:
        print("Error converting data")
        return

    df = df.fillna(method = 'backfill')

    return df


@click.command()
@click.argument("path", type=click.Path(exists=True), default="data/raw/dataset.txt")
def cli_read_data(path: str):
    click.echo('Reading data')
    """
    Function to read csv into pandas DataFrame
    :param path: Path to dataset in csv format
    """
    read_data(path)


if __name__ == "__main__":
    cli_read_data()
