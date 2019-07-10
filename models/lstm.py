from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Reshape, Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from models.base_model import BaseModel


class LstmModel(BaseModel):
    name = "LSTM"

    def build_model(self) -> Model:
        model = Sequential()
        model.add(Reshape((1, self.x_train.shape[1])))
        model.add(LSTM(3, activation='relu'))
        model.add(RepeatVector(1))
        model.add(LSTM(5, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def fit(self):
        ea = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        self.model = KerasRegressor(build_fn=self.build_model, epochs=500,
                                    batch_size=16,
                                    verbose=0,
                                    shuffle=1,
                                    callbacks=[ea])

        self.model.fit(self.x_train, self.y_train)
