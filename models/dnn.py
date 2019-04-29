from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from numpy import ndarray
from sklearn.metrics import mean_squared_error

from models.model_utils import plot_model_loss


class DnnModel:
    SAVED_MODEL_PATH: str = "full_model.h5"

    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = Sequential()
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
        print("New DNN model initialized")

        self.ea = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    def fit(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=150, batch_size=64, verbose=1,
                                 shuffle=1,
                                 callbacks=[self.ea],
                                 validation_data=(self.x_test, self.y_test)).history
        plot_model_loss(history, 'DNN')

    def print_mse(self):
        y_predicted = self.model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_predicted)
        print("Test MSE = ", mse)
