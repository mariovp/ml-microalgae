import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_csv(_filename):
    return pd.read_csv(_filename)


def get_measured_carbohydrates_range_data(_dataframe):
    _last_index = _dataframe[['Carbohydrates']].last_valid_index()
    return _dataframe[0:_last_index + 1]


def fill_interpolable_data(_dataframe):
    _dataframe.loc[:, 'Biomass'] = _dataframe.loc[:, 'Biomass'].interpolate()
    _dataframe.loc[:, 'Carbohydrates'] = _dataframe.loc[:, 'Carbohydrates'].interpolate()
    _dataframe.loc[:, 'Cyanobacteria'] = _dataframe.loc[:, 'Cyanobacteria'].interpolate()
    _dataframe.loc[:, 'Green algae'] = _dataframe.loc[:, 'Green algae'].interpolate()
    _dataframe.loc[:, 'Diatom'] = _dataframe.loc[:, 'Diatom'].interpolate()
    _dataframe.loc[:, 'Protozoa'] = _dataframe.loc[:, 'Protozoa'].interpolate()
    return _dataframe


def fill_non_interpolable_data(_dataframe):
    return _dataframe.fillna(0)


def slide_window(_dataframe):
    _size = _dataframe.shape[0]
    _dataframe = _dataframe.drop(columns=['day'])
    _x = _dataframe.loc[0:_size - 2, :]
    _y = _dataframe.loc[1:_size, ['Carbohydrates']]
    return _x, _y


def normalize(_dataframe):
    _scaler = MinMaxScaler()
    _scaler.fit(_dataframe)
    _array_scaled = _scaler.transform(_dataframe)
    return _array_scaled, _scaler

