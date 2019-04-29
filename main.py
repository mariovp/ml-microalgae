from sklearn.model_selection import train_test_split

import models.preprocessing as pp
from models.cnn_1d import Cnn1DModel

dataframe = pp.read_csv('data/6dhrt.csv')
dataframe = pp.get_measured_carbohydrates_range_data(dataframe)
dataframe = pp.fill_interpolable_data(dataframe)
dataframe = pp.fill_non_interpolable_data(dataframe)

x, y = pp.slide_window(dataframe)

x_normalized, _ = pp.normalize(x)
y_normalized, _ = pp.normalize(y)

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y_normalized, test_size=0.30, random_state=1337)

cnn = Cnn1DModel(x_train, y_train, x_test, y_test)
cnn.fit()
cnn.print_mse()
