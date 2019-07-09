import random

from keras import Model
from keras.callbacks import EarlyStopping, History
from keras.layers import Dense
from keras.models import Sequential

from models.base_model import BaseModel


class AnnModel(BaseModel):
    name = 'ANN'

    def build_model(self) -> Model:
        model = Sequential()
        model.add(Dense(2, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def fit(self) -> History:
        random.seed(1337)
        ea = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        return self.model.fit(self.x_train, self.y_train,
                              epochs=1500,
                              batch_size=16,
                              verbose=1,
                              shuffle=1,
                              callbacks=[ea]).history
