from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel
from models.model_data import ModelData


class RandomForest(BaseModel):
    name = 'Random Forest'

    def __init__(self, model_data: ModelData) -> None:
        super().__init__(model_data)
        self.model = self.build_model()

    def build_model(self):
        return RandomForestRegressor(n_estimators=150, random_state=0)

    def fit(self):
        self.model.fit(self.x_train, self.reshape_y_model(self.y_train))

    def reshape_y_model(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0]))

    def reshape_y_original(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], 1))
