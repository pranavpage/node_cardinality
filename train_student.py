import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Layer, Lambda, InputLayer
from tensorflow.keras.metrics import mean_squared_error as mse 
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
import csv
import matplotlib.pyplot as plt 
from test_srcs import split_data, Distiller, student_info
fname = "./data/train_two_l50_j5_n15000_r0.csv"
data = np.genfromtxt(fname, delimiter=',', skip_header=2000)
np.random.shuffle(data)
split = 0.9
num_samples = data.shape[0]
split_sample = int(split*num_samples)
print(data.shape)
X = data[:split_sample, :-1]
y = data[:split_sample, -1]
X_test = data[split_sample:, :-1]
y_test = data[split_sample:, -1]
y_test = np.reshape(y_test, (num_samples-split_sample, 1))
feature_vec_length = X.shape[1] - 1


# student = Sequential()
# student.add(InputLayer(input_shape=(feature_vec_length, )))
# student.add(Lambda(student_info, output_shape = (feature_vec_length, )))
# student.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
# student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
# student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
# student.add(Dense(1, activation='linear'))