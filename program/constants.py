# https://github.com/PotatoThanh/Bidirectional-LSTM-and-Convolutional-Neural-Network-For-Temperature-Prediction

DATA = '../csv_files/weather.csv'
LEARNING_RATE = 1e-4
WINDOW_SIZE = 8
FEATURE_SIZE = 3

NUM_TRAIN = 15000
NUM_VAL = 1000
BATCH_SIZE = 100
EPOCHS = 10
DELETE_TRAIN = False
DELETE_TEST = False