from keras.models import Model
from keras.models import load_model
from keras.datasets import mnist
from keras import backend as K
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

encoder_path = 'model_encoder.hdf5'
decoder_path = 'model_decoder.hdf5'
autoencoder_path = 'model_autoencoder.hdf5'

# prepare input data
(x_train, _), (x_test, y_test) = mnist.load_data()

# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# load model
encoder = load_model(encoder_path)
decoder = load_model(decoder_path)
autoencoder = load_model(autoencoder_path)

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print('MSQE = ', mean_squared_error(x_test, decoded_imgs))

n = 10  # how many digits we will display
plt.figure(figsize=(10, 2), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

    # display encoded image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8,4))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

plt.show()

K.clear_session()
