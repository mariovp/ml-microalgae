from keras import Model
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from numpy import ndarray

from models.base_model import BaseModel


class Cnn1DModel(BaseModel):
    name = "CNN 1D"

    def build_model(self) -> Model:
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def fit(self) -> History:
        ea = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        return self.model.fit(self.reshape_x_model(self.x_train), self.y_train,
                              epochs=500,
                              batch_size=16,
                              verbose=1,
                              shuffle=1,
                              callbacks=[ea]).history

    def reshape_x_model(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1], 1))

    def reshape_x_original(self, x_array: ndarray) -> ndarray:
        return x_array.reshape((x_array.shape[0], x_array.shape[1]))
