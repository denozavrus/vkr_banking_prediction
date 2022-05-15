from joblib import load
import click
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def predict_logit(x_test, y_test=None):
    try:
        logit = load("models/logit.joblib")
    except:
        print("Model not found")
        return

    try:
        predictions = logit.predict(x_test)
    except:
        print('Error working with model')
        return

    if not (y_test is None):
        plot_confusion_matrix(logit, x_test, y_test)
        plt.show()

        y_score = logit.decision_function(x_test)

        display = PrecisionRecallDisplay.from_predictions(y_test, y_score, name="Logit")
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        plt.show()

        print(
            f'F1 score for logit model is {f1_score(y_test, predictions, average="weighted")}'
        )
    return


@click.command()
@click.argument('x_test')
@click.argument('y_test', default=None)
def cli_predict_logit(x_test, y_test):
    """
    Function to get predictions with logit model
    :param x_test: DataFrame or array with test data
    :param y_test: optional answers for test data
    """
    predict_logit(x_test, y_test)


if __name__ == "__main__":
    cli_predict_logit()
