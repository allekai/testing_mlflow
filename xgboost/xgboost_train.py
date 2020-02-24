# Load training and test datasets
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow import log_metric

if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, 2:]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)


    # Train and save an XGBoost model
    xgb_model = xgb.train(params={'max_depth': 10}, dtrain=dtrain, num_boost_round=10)

    # Evaluate the model
    import pandas as pd
    test_predictions = xgb_model.predict(xgb.DMatrix(x_test))


    log_metric('acc', accuracy_score(y_test, predictions))
