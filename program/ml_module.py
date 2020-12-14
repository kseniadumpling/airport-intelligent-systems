import rdflib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
from sklearn.cluster import KMeans
import constants as cnst
import random


def call_nn(nn, res):
    g = rdflib.Graph()
    g.load("../knowledge_base.n3", format="n3")

    if nn == 'weather_nn':
        res.append(predict_wind_direction())
        res.append(predict_wind_speed())
    else:
        raise ValueError('Undefined neural_network: {}'.format(nn))

# ---------- aircraft ml ----------

def predict_fields(aircraft, faulted_field_list):
    # Clasterization of aircrafts by k-means method, learning without teachers

    dataset = pd.read_csv('../csv_files/aircraft.csv').drop('priority', axis=1).drop('emergency', axis=1)

    num_of_clusters = len(dataset['model'].unique())

    kmeans_model = KMeans(n_clusters=num_of_clusters, max_iter=5)
    kmeans_model.fit(dataset)
    

    default_aircraft = {
        "name": "Default_Name",
        "model": 1,
        "fuel_percent": 50,
        "priority": 5,
        "emergency": 0,
        "airtime": 60
    }

    tmp_aircraft = {}
    for key in default_aircraft:
        if key in aircraft:
            tmp_aircraft[key] = aircraft[key]
        else:
            tmp_aircraft[key] = default_aircraft[key]

    
    test_data = np.array([tmp_aircraft['model'], tmp_aircraft['fuel_percent'], tmp_aircraft['airtime']]).reshape(1, -1)

    # predict and get the aircraft with most suitable field values
    cluster_idx = kmeans_model.predict(test_data)
    res = kmeans_model.cluster_centers_[cluster_idx].reshape(-1)

    if 'model' not in aircraft or 'model' in faulted_field_list:
        tmp_aircraft['model'] = int(round(res[0]))
    if 'fuel_percent' not in aircraft or 'fuel_percent' in faulted_field_list:
        tmp_aircraft['fuel_percent'] = res[1]
    if 'airtime' not in aircraft or 'airtime' in faulted_field_list:
        tmp_aircraft['airtime'] = res[2]

    return tmp_aircraft

# ---------- weather nn ----------
"""
def get_model(input_shape):

    inputs = Input(input_shape)
    inputs = Flatten()(inputs)

    net = Dense(units=50, activation='relu')(inputs)
    net = Dropout(0.3)(net)
    net = Dense(units=30, activation='relu')(net)
    net = Dropout(0.3)(net)
    net = Dense(units=10, activation='relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(units=5, activation='relu')(net)
    net = Dropout(0.1)(net)

    outputs = Dense(units=1)(net)

    return Model(inputs=inputs, outputs=outputs)

def get_data(file_path):
    # IMPORT DATA
    data = []

    data = pd.read_csv(file_path)

    # data table
    data = np.array(data)
    print(len(data))


    # normalize temperature
    temper = data[:, 0]

    temper_min = np.min(temper)
    temper_max = np.max(temper)
    temper = (temper - temper_min)/(temper_max-temper_min)
    data[:, 0] = temper

    # nomalize wind_speed
    wind_sp = data[:, 1]

    wind_sp_min = np.min(wind_sp)
    wind_sp_max = np.max(wind_sp)
    wind_sp = (wind_sp-wind_sp_min)/(wind_sp_max-wind_sp_min)
    data[:, 1] = wind_sp

    # nomalize wind_direction
    wind_dr = data[:, 2]

    wind_dr_min = np.min(wind_dr)
    wind_dr_max = np.max(wind_dr)
    wind_dr = (wind_dr-wind_dr_min)/(wind_dr_max-wind_dr_min)
    data[:, 2] = wind_dr

    # data generator
    features = []
    predict = []
    for i in range(len(data) - cnst.WINDOW_SIZE):
        x = np.array(data[i:i + cnst.WINDOW_SIZE, :]).flatten()
        y = data[i+cnst.WINDOW_SIZE, 0]

        features.append(x)
        predict.append(y)

    features = np.array(features)
    predict = np.array(predict)

    print(features[0])

    # train samples
    x_train = np.reshape(features[0:cnst.NUM_TRAIN, :], (cnst.NUM_TRAIN, cnst.WINDOW_SIZE, cnst.FEATURE_SIZE))
    y_train = np.reshape(predict[0:cnst.NUM_TRAIN], (cnst.NUM_TRAIN, -1))

    # validation samples
    x_val = np.reshape(features[cnst.NUM_TRAIN: cnst.NUM_TRAIN+cnst.NUM_VAL, :], (cnst.NUM_VAL, cnst.WINDOW_SIZE, cnst.FEATURE_SIZE))
    y_val = np.reshape(predict[cnst.NUM_TRAIN: cnst.NUM_TRAIN+cnst.NUM_VAL], (cnst.NUM_VAL, -1))

    # test samples
    NUM_TEST = len(predict) - cnst.NUM_TRAIN - cnst.NUM_VAL
    x_test = np.reshape(features[cnst.NUM_TRAIN+cnst.NUM_VAL: len(predict), :], (NUM_TEST, cnst.WINDOW_SIZE, cnst.FEATURE_SIZE))
    y_test = np.reshape(predict[cnst.NUM_TRAIN+cnst.NUM_VAL: len(predict)], (NUM_TEST, -1))

    return (x_train, y_train, x_val, y_val, y_test, x_test, temper_min, temper_max, NUM_TEST)

def predict_wind_direction():
    x_train, y_train, x_val, y_val, y_test, x_test, temper_min, temper_max, NUM_TEST = get_data('../csv_files/weather.csv')

    input_shape =(cnst.WINDOW_SIZE, cnst.FEATURE_SIZE)

    model = get_model(input_shape)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.mean_squared_error)
    
    model.fit(x_train, y_train, batch_size=cnst.BATCH_SIZE, shuffle=True,
          epochs=cnst.EPOCHS, validation_data=(x_val, y_val))

    test_predict = model.predict(x_test, verbose=1, batch_size=cnst.BATCH_SIZE)
    test_predict = np.reshape(test_predict, (NUM_TEST, -1))

    # become original temperature
    temper_predict = np.array(test_predict)*(temper_max - temper_min) + temper_min

    print(temper_predict)
    return temper_predict
"""

def predict_wind_direction():
    # todo: change this emulation to the real NN prediction once it's not have bugs 
    rand_deg = random.randint(0,359)
    
    if 0 <= rand_deg < 45 or 315 <= rand_deg <= 359:
        return 'n'
    elif 45 <= rand_deg < 135:
        return 'e'
    elif 135 <= rand_deg < 225:
        return 's'
    elif 225 <= rand_deg < 315:
        return 'w'
    else: 
        raise ValueError('Undefined direction: {}'.format(rand_deg))

def predict_wind_speed():
    # todo: change this emulation to the real NN prediction once it's not have bugs 
    return random.randint(0,20)




if __name__ == "__main__":
    print('nn_module.py file')