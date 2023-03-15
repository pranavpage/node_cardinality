import numpy as np 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import csv
import matplotlib.pyplot as plt 

# import data first 
fname = "./data/final_l50_j5_n50000.csv"
data = np.genfromtxt(fname, delimiter=',')
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
feature_vec_length = X.shape[1]

model = Sequential()
model.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
model.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
model.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X,y, validation_data=(X_test, y_test), epochs = 600, batch_size = 64, shuffle=True)
model.save("./models/model_two_l50_j5_n50000_r0")
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0, 1e-4)
plt.grid()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./plots/final_loss.png")
y_predict = model.predict(X_test)
y_test = np.reshape(y_test, (num_samples-split_sample, 1))
stddev = (((y_predict-y_test)**2).mean(axis = 0))**0.5

bnb_stddev = (((np.reshape(X_test[:, -1], (num_samples-split_sample, 1)) - y_test)**2).mean(axis=0))**0.5
print(f"Average test mse = {stddev**2}")
print(f"Average BnB mse = {bnb_stddev**2}")
print(f"unseen bitmaps")
for i in range(10):
    print(f"Actual = {int(y_test[i,0]*256):d}, Predicted = {y_predict[i,0]*256:.2f}, bnb estimate={X_test[i,-1]*256:.2f}")