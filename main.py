import os

import numpy as np
from sklearn.model_selection import train_test_split

import models.preprocessing as pp
from models.ann import AnnModel
from models.cnn_1d import Cnn1DModel
from models.knn import KnnModel
from models.lstm import LstmModel
from models.model_data import ModelData
from models.model_utils import plot_comparison
from models.random_forest import RandomForest

path = 'data/'

files = os.listdir(path)

x = list()
y = list()

filepath = ''

for file in files:
    filepath = os.path.join(path, file)
    dataframe = pp.read_csv(filepath)
    dataframe = pp.get_measured_carbohydrates_range_data(dataframe)
    dataframe = pp.fill_interpolable_data(dataframe)
    dataframe = pp.fill_non_interpolable_data(dataframe)
    xn, yn = pp.slide_window(dataframe)
    x.extend(xn.values)
    y.extend(yn.values)

feature_names = pp.get_feature_names(filepath)
print(feature_names)
print(type(feature_names))

x = np.array(x, dtype='float64')
y = np.array(y, dtype='float64')

x_normalized, x_scaler = pp.normalize(x)
y_normalized, y_scaler = pp.normalize(y)

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_normalized, test_size=0.25, random_state=1337)

model_data = ModelData(x_train, x_test, y_train, y_test, x_scaler, y_scaler, feature_names)

ann = AnnModel(model_data)
cnn = Cnn1DModel(model_data)
lstm = LstmModel(model_data)
knn = KnnModel(model_data)
rf = RandomForest(model_data)

models = list()

models.append(ann)
models.append(cnn)
models.append(lstm)
models.append(knn)
models.append(rf)

for model in models:
    model.train()

for model in models:
    model.evaluate()

for model in models:
    model.show_feature_importance()

plot_comparison(ann, cnn, lstm, knn, rf)
