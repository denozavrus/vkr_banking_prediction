import pandas as pd
import click
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame):

    x, y = df.loc[:, df.columns != "loan_type"], df.loc[:, "loan_type"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=42
    )

    ros = RandomOverSampler(sampling_strategy=0.4, random_state=42)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    ros = RandomOverSampler(sampling_strategy=0.4, random_state=42)
    x_test, y_test = ros.fit_resample(x_test, y_test)

    return x_train, y_train, x_test, y_test


@click.command()
@click.argument("df")
def cli_split_data(df: pd.DataFrame):
    """
    Splits dataset into two parts
    :param df: pandas DataFrame with data
    :return: Returns x_train, x_test and y_train, y_test where x is features and y is target
    """
    split_data(df)


if __name__ == "__main__":
    cli_split_data()
