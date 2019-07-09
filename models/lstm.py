import random

from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from numpy import ndarray

from models.base_model import BaseModel


class LstmModel(BaseModel):
    name = "LSTM"

    def build_model(self) -> Model:
        model = Sequential()
        model.add(LSTM(3, activation='relu'))
        model.add(RepeatVector(1))
        model.add(LSTM(5, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
        return model

    def fit(self):
        random.seed(1337)
        ea = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        return self.model.fit(self.reshape_x_model(self.x_train), self.reshape_y_model(self.y_train),
                              epochs=500,
                              batch_size=16,
                              verbose=1,
                              shuffle=1,
                              callbacks=[ea]).history

    def reshape_x_model(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1], 1))

    def reshape_y_model(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], y_array.shape[1], 1))

    def reshape_x_original(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1]))

    def reshape_y_original(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], y_array.shape[1]))
