import src
import click
import warnings


warnings.filterwarnings("ignore")

@click.command()
@click.option('--train', '--t', default=False, is_flag=True)
@click.option('--eval', '--e', default=False, is_flag=True)
@click.argument('path', default='data/raw/dataset.txt')
def cli(train, eval, path):
    import os
    print(os.getcwd())

    df = src.read_data(path)
    df = src.build_features(df)

    x_train, y_train, x_test, y_test = src.split_data(df)
    x_train, x_test = src.one_hot_encode (x_train, x_test)
    x_train, x_test = src.scale_data(x_train, x_test)

    if train:
        src.train_logit(x_train, y_train.values)

    if eval:
        src.predict_logit(x_test, y_test.values)


if __name__ == '__main__':
    cli()

