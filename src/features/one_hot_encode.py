import pandas as pd
import click
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(x_train: pd.DataFrame, x_test: pd.DataFrame):

    enc = OneHotEncoder(handle_unknown="ignore")
    enc_features = enc.fit_transform(x_train[["gender"]]).toarray()

    enc1 = OneHotEncoder(handle_unknown="ignore")
    enc_features1 = enc1.fit_transform(x_train[["card_type"]]).toarray()

    x_train = pd.concat(
        [
            x_train,
            pd.DataFrame(enc_features, columns=["f", "m"]),
            pd.DataFrame(enc_features1),
        ],
        axis=1,
    )

    x_train.drop(["gender", "card_type"], axis=1, inplace=True)

    enc_features = enc.transform(x_test[["gender"]]).toarray()
    enc_features1 = enc1.transform(x_test[["card_type"]]).toarray()

    x_test.drop(["gender", "card_type"], axis=1, inplace=True)

    x_test = pd.concat(
        [
            x_test,
            pd.DataFrame(enc_features, columns=["f", "m"]),
            pd.DataFrame(enc_features1),
        ],
        axis=1,
    )

    return x_train, x_test


@click.command()
@click.argument('x_train')
@click.argument('x_test')
def cli_one_hot_encode(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """
    Function to one hot encode categorical features
    :param x_train: Train part of dataset
    :param x_test: Test part of dataset
    """
    one_hot_encode(x_train, x_test)


if __name__ == '__main__':
    cli_one_hot_encode()
