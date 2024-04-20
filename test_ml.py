import pytest
from ml.data import process_data
from ml.model import compute_model_metrics
from sklearn.linear_model import LogisticRegression
from ml.model import train_model
import pandas as pd
import numpy as np

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_model_uses_logistic_regression():
    """
    Test if the model training function uses Logistic Regression.
    """
    sample_X = np.array([[0, 1], [1, 0]])  # Dummy feature data
    sample_y = np.array([0, 1])  # Dummy target data
    model = train_model(sample_X, sample_y)
    assert isinstance(model, LogisticRegression), "Model should be an instance of Logistic Regression"
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_metric_computation_accuracy():
    """
    Test if the metric computation returns expected values.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 0.5, "Expected precision to be 0.5"
    assert recall == 0.5, "Expected recall to be 0.5"
    assert fbeta == 0.5, "Expected F1 score to be 0.5"
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_data_processing_returns_correct_type():
    """
    Test if the data processing function returns the correct types.
    """
    # Assuming there's a function to generate or load a sample dataset
    sample_data = {'age': [25, 30], 'workclass': ['Private', 'State-gov']}
    categorical_features = ['workclass']
    label = 'age'

    X, y, _, _ = process_data(pd.DataFrame(sample_data), categorical_features, label, training=True)
    assert isinstance(X, np.ndarray), "Expected X to be a numpy array"
    assert isinstance(y, np.ndarray), "Expected y to be a numpy array"
    pass
