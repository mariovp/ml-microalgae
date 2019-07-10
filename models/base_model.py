from abc import ABC, abstractmethod

import eli5
import numpy as np
from eli5.sklearn import PermutationImportance
from numpy import ndarray
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.model_data import ModelData
from models.model_utils import plot_model_loss, plot_real_vs_predicted_values, plot_bokeh, metric_to_string


class BaseModel(ABC):

    def __init__(self, model_data: ModelData) -> None:
        super().__init__()

        self.x_train = model_data.x_train
        self.x_test = model_data.x_test
        self.y_train = model_data.y_train
        self.y_test = model_data.y_test
        self.x_scaler = model_data.x_scaler
        self.y_scaler = model_data.y_scaler
        self.feature_names = model_data.feature_names

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

    def train(self, show_history=False):
        np.random.seed(1)  # for reproducibility
        history = self.fit()
        if show_history and history is not None:
            self.plot_model_loss(history)

    def predict(self, x_predict: ndarray) -> ndarray:
        y_predicted = self.model.predict(x_predict)
        y_predicted = y_predicted.reshape(y_predicted.shape[0], 1)
        return y_predicted

    def plot_model_loss(self, history):
        plot_model_loss(history, self.name)

    def plot_real_vs_predicted_normalized(self, mse):
        y_predicted = self.predict(self.x_test)
        plot_real_vs_predicted_values(self.y_test, y_predicted, self.name, mse)

    def plot_real_vs_predicted(self, mse):
        y_predicted = self.predict(self.x_test)
        y_predicted_descaled = self.y_scaler.inverse_transform(y_predicted)
        y_real_descaled = self.y_scaler.inverse_transform(self.y_test)
        plot_bokeh(y_real_descaled, y_predicted_descaled, self.name, mse)

    def get_mse(self):
        y_predicted = self.predict(self.x_test)
        return mean_squared_error(self.y_test, y_predicted)

    def evaluate(self):
        y_predicted = self.predict(self.x_test)

        mse = mean_squared_error(self.y_test, y_predicted)
        mae = mean_absolute_error(self.y_test, y_predicted)
        r2 = r2_score(self.y_test, y_predicted)

        print()
        print("<Model Evaluation>")
        print(self.name)
        print("MSE = " + metric_to_string(mse))
        print("MAE = " + metric_to_string(mae))
        print("R2 Score = " + metric_to_string(r2))

        self.plot_real_vs_predicted(mse)

    def get_eli5_model(self):
        return self.model

    def show_feature_importance(self):
        print("\n\n\n\n+++++++++++++++++++++++++++++++")
        print('Calculating feature importance for model ', self.name, "...")
        perm = PermutationImportance(self.get_eli5_model(), random_state=1).fit(self.x_test, self.y_test)
        print(self.name, 'model feature importance')
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=self.feature_names)))

