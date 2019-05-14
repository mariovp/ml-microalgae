from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
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
        model.compile(Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
        return model

    def fit(self):
        ea = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        return self.model.fit(self.reshape_x_model(self.x_train), self.reshape_y_model(self.y_train),
                              epochs=500,
                              batch_size=16,
                              verbose=1,
                              shuffle=1,
                              callbacks=[ea],
                              validation_split=0.1).history

    def reshape_x_model(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1], 1))

    def reshape_y_model(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], y_array.shape[1], 1))

    def reshape_x_original(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1]))

    def reshape_y_original(self, y_array: ndarray) -> ndarray:
        return y_array.reshape((y_array.shape[0], y_array.shape[1]))
