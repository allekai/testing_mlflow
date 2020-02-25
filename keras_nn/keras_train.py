from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow import log_metric
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.keras import log_model
import mlflow
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    remote_server_uri = 'http://127.0.0.1:1234'  # set to your server URI

    # Make sure to also set environment variable MLFLOW_TRACKING_URI='remote_server_uri'
    # see: https://github.com/mlflow/mlflow/issues/608#issuecomment-454316004
    mlflow.set_tracking_uri(remote_server_uri)
    conda_env = 'keras.yaml'

    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # One hot encoding
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()

    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        model = tf.keras.Sequential([
          tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  
          tf.keras.layers.Dense(10, activation=tf.nn.relu),
          tf.keras.layers.Dense(3, activation="softmax")
        ])
        model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1000)
        test_loss, test_acc = model.evaluate(x_test,  y_test)

        log_metric('acc', test_acc)
        log_model(keras_model = model,
                registered_model_name = 'Keras-Iris-Model',
                artifact_path = 'model_artifact',
                conda_env = conda_env)

