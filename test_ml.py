import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier

# Test 1: Check if train_model returns a RandomForestClassifier instance
def test_train_model():
    """
    Test if train_model correctly returns a RandomForestClassifier instance.
    """
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "train_model did not return a RandomForestClassifier instance."

# Test 2: Check if compute_model_metrics returns correct metric values
def test_compute_model_metrics():
    """
    Test if compute_model_metrics correctly computes precision, recall, and F1-score.
    """
    y = np.array([1, 0, 1, 1, 0, 0, 1])
    preds = np.array([1, 0, 1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0, "Precision is incorrect."
    assert recall == 0.75, "Recall is incorrect."
    assert np.isclose(fbeta, 0.8571, atol=0.0001), "F1-score is incorrect."

# Test 3: Check if inference returns predictions of expected shape
def test_inference():
    """
    Test if inference correctly returns predictions of the expected shape.
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == (10,), "Inference output shape is incorrect."