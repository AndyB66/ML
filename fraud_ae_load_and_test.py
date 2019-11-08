import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

train_data_path = 'fraud_train.pickle'
test_data_path = 'fraud_test.pickle'
encoder_path = 'fraud_encoder.hdf5'
decoder_path = 'fraud_decoder.hdf5'
autoencoder_path = 'fraud_autoencoder.hdf5'

# prepare input data
x_train = pd.read_pickle(train_data_path)
x_test = pd.read_pickle(test_data_path)

# load model
encoder = load_model(encoder_path)
decoder = load_model(decoder_path)
autoencoder = load_model(autoencoder_path)

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# encode and decode some data
# note that we take them from the *test* set
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

print('MSQE = ', mean_squared_error(x_test, decoded_data))

# Visualize the reconstructed encoded representations
#dim = x_test.shape[1]
 
n = 10  # how many events we will display
plt.figure(figsize=(20, 20), dpi=300)
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test.iloc[i].values.reshape(3, 13))
    plt.gray()
    ax.set_axis_off()

    # display encoded data
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_data[i].reshape(1, 3))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_data[i].reshape(3, 13))
    plt.gray()
    ax.set_axis_off()

plt.show()

K.clear_session()
