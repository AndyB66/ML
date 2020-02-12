from ae_lib import iRprop_
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

decoder_path = 'fraud_decoder_deep_optimal.hdf5'
autoencoder_path = 'fraud_autoencoder_deep_optimal.hdf5'
encoding_dim = 3

# load model
autoencoder = load_model(autoencoder_path, custom_objects={'iRprop_': iRprop_})

# create a placeholder for an encoded (3-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last two layers of the autoencoder model
decoder_layer_1 = autoencoder.layers[-2]
decoder_layer_2 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer_2(decoder_layer_1(encoded_input)))

# configure model
decoder.compile(optimizer=iRprop_(), loss='mean_squared_error')
autoencoder.compile(optimizer=iRprop_(), loss='mean_squared_error')

# save model
decoder.save(decoder_path)

K.clear_session()
