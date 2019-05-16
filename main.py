import os

import numpy as np
from sklearn.model_selection import train_test_split

import models.preprocessing as pp
from models.ann import AnnModel
from models.cnn_1d import Cnn1DModel
from models.decision_tree import DecisionTree
from models.knn import KnnModel
from models.model_data import ModelData
# random.seed(1337)
from models.random_forest import RandomForest

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

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_normalized, test_size=0.25, random_state=1337)

model_data = ModelData(x_train, x_test, y_train, y_test, x_scaler, y_scaler)

ann = AnnModel(model_data)
ann.train()

cnn = Cnn1DModel(model_data)
cnn.train()

knn = KnnModel(model_data)
knn.grid_search_optimization()

dt = DecisionTree(model_data)
dt.train()

rf = RandomForest(model_data)
rf.train()

"""
lstm = LstmModel(model_data)
lstm.train()
"""
