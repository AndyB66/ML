import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.losses import mean_squared_error, binary_crossentropy
from ae_lib import iRprop_

train_data_path = 'fraud_train_optimal.pickle'
test_data_path = 'fraud_test_optimal.pickle'
encoder_path = 'fraud_encoder_deep_optimal.hdf5'
decoder_path = 'fraud_decoder_deep_optimal.hdf5'
autoencoder_path = 'fraud_autoencoder_deep_optimal.hdf5'

# prepare input data
x_train = pd.read_pickle(train_data_path)
x_test = pd.read_pickle(test_data_path)

# load model
encoder = load_model(encoder_path, custom_objects={'iRprop_': iRprop_})
decoder = load_model(decoder_path, custom_objects={'iRprop_': iRprop_})
autoencoder = load_model(autoencoder_path, custom_objects={'iRprop_': iRprop_})

# configure model
encoder.compile(optimizer=iRprop_(), loss='mean_squared_error')
decoder.compile(optimizer=iRprop_(), loss='mean_squared_error')
autoencoder.compile(optimizer=iRprop_(), loss='mean_squared_error')

# encode and decode some data
# note that we take them from the *test* set
encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

print('MSQE = ', K.eval(K.mean(mean_squared_error(tf.convert_to_tensor(x_test), tf.convert_to_tensor(decoded_data.astype(np.float64))))))
print('XENT = ', K.eval(K.mean(binary_crossentropy(tf.convert_to_tensor(x_test), tf.convert_to_tensor(decoded_data.astype(np.float64))))))

# Visualize the reconstructed encoded representations

n = 10  # how many events we will display
plt.figure(figsize=(20, 20), dpi=300)
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test.iloc[i].values.reshape(3, 11))
    plt.gray()
    ax.set_axis_off()

    # display encoded data
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_data[i].reshape(1, 3))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_data[i].reshape(3, 11))
    plt.gray()
    ax.set_axis_off()

plt.show()

K.clear_session()
