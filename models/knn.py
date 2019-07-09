from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

from models.base_model import BaseModel


class KnnModel(BaseModel):

    name = 'KNN'

    def build_model(self):
        return KNeighborsRegressor(8, weights='uniform')

    def fit(self):
        self.grid_search_optimization()

    def grid_search_optimization(self):
        optimization_log = list()
        model_count = 0

        for i, weights in enumerate(['uniform', 'distance']):
            for k in range(2, 21):
                model_count += 1
                self.model = neighbors.KNeighborsRegressor(k, weights=weights)
                self.model.fit(self.x_train, self.y_train)
                mse = self.get_mse()
                optimization_log.append([mse, k, weights])

        print("Grid search trained " + str(model_count) + " KNN models")

        optimization_log.sort(key=lambda x: x[0])
        best_model_params = optimization_log[0]
        best_mse = best_model_params[0]
        best_k = best_model_params[1]
        best_weights = best_model_params[2]

        print("> Best Model <")
        print("MSE = " + str(round(best_mse, 4)))
        print("Parameters: k = " + str(best_k) + ", weights = " + best_weights)

        print("Re-training with best parameters for test")
        self.model = neighbors.KNeighborsRegressor(best_k, weights=best_weights)
        self.model.fit(self.x_train, self.y_train)
