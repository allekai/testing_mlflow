from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from mlflow import log_metric

if __name__ == "__main__":
    iris= datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(C=1e5)

    logreg.fit(x_train, y_train)

    predictions = logreg.predict(x_test)

    log_metric('acc', accuracy_score(y_test, predictions))

