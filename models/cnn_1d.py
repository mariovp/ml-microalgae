from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Reshape
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from models.base_model import BaseModel


class Cnn1DModel(BaseModel):
    name = "CNN 1D"

    def build_model(self) -> Model:
        model = Sequential()
        model.add(Reshape((self.x_train.shape[1], 1)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def fit(self):
        ea = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        self.model = KerasRegressor(build_fn=self.build_model, epochs=400,
                                    batch_size=16,
                                    verbose=0,
                                    shuffle=1,
                                    callbacks=[ea])
        self.model.fit(self.x_train, self.y_train)

