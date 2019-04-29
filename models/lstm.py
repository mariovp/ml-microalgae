from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from numpy import ndarray
from sklearn.metrics import mean_squared_error

from models.model_utils import plot_model_loss, plot_real_vs_predicted_values


class LstmModel:

    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray):
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        self.y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        self.y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

        self.model = Sequential()
        self.model.add(LSTM(3, activation='relu'))
        self.model.add(RepeatVector(1))
        self.model.add(LSTM(5, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
        self.ea = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def fit(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=500, batch_size=16, verbose=1,
                                 shuffle=1,
                                 callbacks=[self.ea],
                                 validation_split=0.1).history
        mse = self.get_mse()
        plot_model_loss(history, 'LSTM', mse)

    def get_mse(self):
        y_predicted = self.model.predict(self.x_test)
        y_predicted = y_predicted.reshape((self.y_test.shape[0], self.y_test.shape[1]))
        return mean_squared_error(self.y_test.reshape((self.y_test.shape[0], self.y_test.shape[1])), y_predicted)

    def plot_real_vs_predicted(self):
        y_predicted = self.model.predict(self.x_test)
        y_predicted = y_predicted.reshape((self.y_test.shape[0], self.y_test.shape[1]))
        plot_real_vs_predicted_values(self.y_test.reshape((self.y_test.shape[0], self.y_test.shape[1])), y_predicted, 'LSTM')
