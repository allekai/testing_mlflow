from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from mlflow import log_metric
from mlflow.sklearn import log_model
import mlflow

if __name__ == "__main__":
    remote_server_uri = 'http://127.0.0.1:1234'  # set to your server URI

    # Make sure to also set environment variable MLFLOW_TRACKING_URI='remote_server_uri'
    # see: https://github.com/mlflow/mlflow/issues/608#issuecomment-454316004
    mlflow.set_tracking_uri(remote_server_uri)
    conda_env = 'log_reg.yaml'

    iris= datasets.load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        logreg = LogisticRegression(C=1e5)
        logreg.fit(x_train, y_train)
        predictions = logreg.predict(x_test)

        log_metric('acc', accuracy_score(y_test, predictions))
        log_model(sk_model = logreg,
                registered_model_name = 'LogisticReg-Iris-Model',
                artifact_path = 'model_artifact',
                conda_env = conda_env)

