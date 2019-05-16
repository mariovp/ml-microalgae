from typing import NamedTuple

from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler


class ModelData(NamedTuple):
    """A data container for models
    Attributes:
    -----------
        x_train: Training data
        x_test: Test data
        y_train: Training targets
        y_test: Test targets
        x_scaler: Scaler to return x data to original domain
        y_scaler: Scaler to return y data to original domain
    """
    x_train: ndarray
    x_test: ndarray
    y_train: ndarray
    y_test: ndarray
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
