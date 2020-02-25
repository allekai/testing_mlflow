from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow import log_metric
from mlflow.xgboost import log_model

import mlflow
import xgboost as xgb


if __name__ == '__main__':
    remote_server_uri = 'http://127.0.0.1:1234'  # set to your server URI

    # Make sure to also set environment variable MLFLOW_TRACKING_URI='remote_server_uri'
    # see: https://github.com/mlflow/mlflow/issues/608#issuecomment-454316004
    mlflow.set_tracking_uri(remote_server_uri)
    conda_env = 'xgboost.yaml'
    
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    
    with mlflow.start_run():
        # Train and save an XGBoost model
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(x_train, y_train)

        test_predictions = xgb_model.predict(x_test)


        log_metric('acc', accuracy_score(y_test, test_predictions))

        log_model(xgb_model=xgb_model,
                registered_model_name='XGBoost-Iris-Model',
                artifact_path='model_artifact',
                conda_env=conda_env)

