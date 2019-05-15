from abc import ABC, abstractmethod

from numpy import ndarray
from sklearn.metrics import mean_squared_error

from models.model_data import ModelData
from models.model_utils import plot_model_loss, plot_real_vs_predicted_values


class BaseModel(ABC):

    def __init__(self, model_data: ModelData) -> None:
        super().__init__()

        self.x_train = model_data.x_train
        self.x_test = model_data.x_test
        self.y_train = model_data.y_train
        self.y_test = model_data.y_test
        self.x_scaler = model_data.x_scaler
        self.y_scaler = model_data.y_scaler

        self.model = self.build_model()

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    def train(self):
        history = self.fit()
        if history is not None:
            self.plot_model_loss(history)
        self.plot_real_vs_predicted()

    def predict(self, x_predict: ndarray) -> ndarray:
        x_predict = self.reshape_x_model(x_predict)
        y_predicted = self.model.predict(x_predict)
        return self.reshape_y_original(y_predicted)

    def plot_model_loss(self, history):
        mse = self.get_mse()
        plot_model_loss(history, self.name, mse)

    def plot_real_vs_predicted(self):
        y_predicted = self.predict(self.x_test)
        plot_real_vs_predicted_values(self.y_test, y_predicted, self.name)

    def reshape_x_model(self, x_array: ndarray) -> ndarray:
        return x_array

    def reshape_y_model(self, y_array: ndarray) -> ndarray:
        return y_array

    def reshape_x_original(self, x_array: ndarray) -> ndarray:
        return x_array

    def reshape_y_original(self, y_array: ndarray) -> ndarray:
        return y_array

    def get_mse(self):
        y_predicted = self.predict(self.reshape_x_model(self.x_test))
        return mean_squared_error(self.y_test, y_predicted)
