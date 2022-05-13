import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import dump


def train_logit(x_train: pd.DataFrame, y_train: pd.Series):

    print('Logit is being trained')
    grid = {
        "C": np.linspace(0.1, 3, 10),
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced"],
        "solver": ["liblinear", "lbfgs", "newton-cg"],
        "max_iter": [1000],
    }  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(x_train, y_train)

    print("tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)

    print('Saving best model')
    dump(logreg_cv.best_estimator_, "models/logit.joblib")

    return


@click.command()
@click.argument("x_train")
@click.argument("y_train")
def cli_train_logit(x_train: pd.DataFrame, y_train: pd.Series):
    """
    Train logit model
    :param x_train: pandas DataFrame with training data
    :param y_train: pandas Series with targets for training
    """
    train_logit(x_train, y_train)


if __name__ == "__main__":
    cli_train_logit()
