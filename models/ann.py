from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from models.base_model import BaseModel


class AnnModel(BaseModel):
    name = 'ANN'

    def build_model(self) -> Model:
        model = Sequential()
        model.add(Dense(2, activation='relu', input_shape=(18,)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def fit(self):
        ea = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        self.model = KerasRegressor(build_fn=self.build_model,
                                    epochs=1500,
                                    batch_size=16,
                                    verbose=0,
                                    shuffle=1,
                                    callbacks=[ea])
        self.model.fit(self.x_train, self.y_train)
