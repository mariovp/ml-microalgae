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
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def fit(self) -> History:
        ea = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        return self.model.fit(self.x_train, self.y_train,
                              epochs=150,
                              batch_size=64,
                              verbose=1,
                              shuffle=1,
                              callbacks=[ea],
                              validation_split=0.1).history
