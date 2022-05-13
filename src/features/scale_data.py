import pandas as pd
import click
from sklearn.preprocessing import StandardScaler


def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame):

    print('Scaling data')
    scaler = StandardScaler()
    columns = x_train.columns
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=columns)

    columns = x_test.columns
    x_test = scaler.transform(x_test)
    x_test = pd.DataFrame(x_test, columns=columns)

    return x_train, x_test


@click.command()
@click.argument("x_train")
@click.argument("x_test")
def cli_scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """
    Scale data Standard Scaler from sklearn
    :param x_train: pandas DataFrame with training data
    :param x_test: pandas DataFrame with testing data
    """
    scale_data(x_train, x_test)


if __name__ == "__main__":
    cli_scale_data()
