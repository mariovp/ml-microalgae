from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from numpy import ndarray
from sklearn.metrics import mean_squared_error

from models.model_utils import plot_model_loss
from models.model_utils import plot_real_vs_predicted_values


class Cnn1DModel:

    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray):
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.y_train = y_train
        self.x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        self.y_test = y_test

        self.model = Sequential()
        self.model.add(Conv1D(filters=30, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        self.ea = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    def fit(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=150, batch_size=16, verbose=1,
                                 shuffle=1,
                                 callbacks=[self.ea],
                                 validation_split=0.1).history
        mse = self.get_mse()
        plot_model_loss(history, 'CNN', mse)

    def get_mse(self):
        y_predicted = self.model.predict(self.x_test)
        return mean_squared_error(self.y_test, y_predicted)

    def plot_real_vs_predicted(self):
        y_predicted = self.model.predict(self.x_test)
        plot_real_vs_predicted_values(self.y_test, y_predicted, 'CNN')
