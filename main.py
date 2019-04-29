import os

from sklearn.model_selection import train_test_split

import models.preprocessing as pp
from models.dnn import DnnModel
from models.cnn_1d import Cnn1DModel

import numpy as np

path = 'data/'

files = os.listdir(path)

x = list()
y = list()

for file in files:
    filepath = os.path.join(path, file)
    dataframe = pp.read_csv(filepath)
    dataframe = pp.get_measured_carbohydrates_range_data(dataframe)
    dataframe = pp.fill_interpolable_data(dataframe)
    dataframe = pp.fill_non_interpolable_data(dataframe)
    xn, yn = pp.slide_window(dataframe)
    x.extend(xn.values)
    y.extend(yn.values)

x = np.array(x, dtype='float64')
y = np.array(y, dtype='float64')

x_normalized, x_scaler = pp.normalize(x)
y_normalized, y_scaler = pp.normalize(y)

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_normalized, test_size=0.30, random_state=1337)

dnn = DnnModel(x_train, y_train, x_test, y_test)
dnn.fit()
dnn.print_mse()

cnn = Cnn1DModel(x_train, y_train, x_test, y_test)
cnn.fit()
cnn.print_mse()
