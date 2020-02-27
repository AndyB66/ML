import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from ae_lib import iRprop_

train_data_path = 'fraud_old_train.pickle'
test_data_path = 'fraud_old_test.pickle'
encoder_path = 'fraud_encoder_deep_old.hdf5'
decoder_path = 'fraud_decoder_deep_old.hdf5'
autoencoder_path = 'fraud_autoencoder_deep_old.hdf5'
msqe_plot_path = 'fraud_ae_msqe_old.png'
xent_plot_path = 'fraud_ae_xent_old.png'

# prepare input data
x_train = pd.read_pickle(train_data_path).to_numpy()
x_test = pd.read_pickle(test_data_path).to_numpy()

# load model
encoder = load_model(encoder_path, custom_objects={'iRprop_': iRprop_})
decoder = load_model(decoder_path, custom_objects={'iRprop_': iRprop_})
autoencoder = load_model(autoencoder_path, custom_objects={'iRprop_': iRprop_})

# configure model
encoder.compile(optimizer=iRprop_(), loss='binary_crossentropy')
decoder.compile(optimizer=iRprop_(), loss='binary_crossentropy')
autoencoder.compile(optimizer=iRprop_(), loss='binary_crossentropy')
#encoder.compile(optimizer=iRprop_(), loss='mean_squared_error')
#decoder.compile(optimizer=iRprop_(), loss='mean_squared_error')
#autoencoder.compile(optimizer=iRprop_(), loss='mean_squared_error')

# encode and decode some data
enc_train = encoder.predict(x_train)
y_train = decoder.predict(enc_train)
enc_test = encoder.predict(x_test)
y_test = decoder.predict(enc_test)

msqe_train = np.mean(np.square(x_train - y_train), axis = -1)
msqe_test = np.mean(np.square(x_test - y_test), axis = -1)

tfs = tf.InteractiveSession()
xent_train = binary_crossentropy(tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train.astype(np.float64))).eval()
xent_test = binary_crossentropy(tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test.astype(np.float64))).eval()

print('MSQE(train, test) = ', np.mean(msqe_train), np.mean(msqe_test))
print('XENT(train, test) = ', np.mean(xent_train), np.mean(xent_test))

# Visualise MSQE/XENT using boxplots
msqe = {'Train': msqe_train, 'Test': msqe_test}
fig1, ax1 = plt.subplots()
plt.title('MSQE')
ax1.boxplot(msqe.values())
ax1.set_xticklabels(msqe.keys())
plt.plot()
plt.savefig(msqe_plot_path)

xent = {'Train': xent_train, 'Test': xent_test}
fig2, ax2 = plt.subplots()
plt.title('XENT')
ax2.boxplot(xent.values())
ax2.set_xticklabels(xent.keys())
plt.plot()
plt.savefig(xent_plot_path)

plt.show()

# Visualize the reconstructed encoded representations
'''
n = 10  # how many events we will display
plt.figure(figsize=(20, 20), dpi=300)
for i in range(n):
    # display original
    ax2 = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(3, 11))
    plt.gray()
    ax2.set_axis_off()

    # display encoded data
    ax2 = plt.subplot(3, n, i + 1 + n)
    plt.imshow(enc_test[i].reshape(1, 3))
    plt.gray()
    ax2.set_axis_off()

    # display reconstruction
    ax2 = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(y_test[i].reshape(3, 11))
    plt.gray()
    ax2.set_axis_off()
plt.savefig(aerec_plot_path)
plt.show()
'''
K.clear_session()
