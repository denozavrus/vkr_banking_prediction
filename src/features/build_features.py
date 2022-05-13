import pandas as pd
import click


def build_features(df: pd.DataFrame):

    print('Preprocessing data')
    df["amount/salary"] = df["amount"] / df["average_salary"]
    df["payments/salary"] = df["payments"] / df["average_salary"]
    df["age * amount"] = df["age"] * df["amount"]
    df["age * salary"] = df["age"] * df["average_salary"]

    def age_group(x):
        if x < 24:
            return 1
        if 24 <= x < 30:
            return 2
        if 30 <= x < 42:
            return 3
        if x >= 42:
            return 4

    def duration_group(x):
        return x // 12

    df["age_group"] = df["age"].apply(age_group)
    df["duration_group"] = df["duration"].apply(duration_group)

    return df


@click.command()
@click.argument("df")
def cli_build_features(df: pd.DataFrame):
    """
    Function for feature engineering creates several new features in df
    :param df: pandas DataFrame dataset
    """
    build_features(df)


if __name__ == "__main__":
    cli_build_features()
