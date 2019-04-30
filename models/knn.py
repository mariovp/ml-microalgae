from numpy import ndarray
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

from models.model_utils import plot_real_vs_predicted_values


class KnnModel:

    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = neighbors.KNeighborsRegressor(8, weights='uniform')

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

        print("Grid search trained "+str(model_count)+" KNN models")

        optimization_log.sort(key=lambda x: x[0])
        best_model_params = optimization_log[0]
        best_mse = best_model_params[0]
        best_k = best_model_params[1]
        best_weights = best_model_params[2]

        print("> Best Model <")
        print("MSE = "+str(round(best_mse, 4)))
        print("Parameters: k = "+str(best_k)+", weights = "+best_weights)

        print("Re-training with best parameters for test")
        self.model = neighbors.KNeighborsRegressor(best_k, weights=best_weights)
        self.model.fit(self.x_train, self.y_train)
        self.plot_real_vs_predicted()

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def get_mse(self):
        y_predicted = self.model.predict(self.x_test)
        return mean_squared_error(self.y_test, y_predicted)

    def plot_real_vs_predicted(self):
        y_predicted = self.model.predict(self.x_test)
        plot_real_vs_predicted_values(self.y_test, y_predicted, 'KNN')
