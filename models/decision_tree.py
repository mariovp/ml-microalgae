from numpy import ndarray
from sklearn.tree import DecisionTreeRegressor

from models.base_model import BaseModel


class DecisionTree(BaseModel):
    name = 'DecisionTree'

    def build_model(self):
        return DecisionTreeRegressor(random_state=0)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def reshape_y_model(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0]))

    def reshape_y_original(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], 1))
