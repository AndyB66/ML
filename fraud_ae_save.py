import pandas as pd
from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

input_data_path = "data-full.csv"
train_data_path = 'fraud_train.pickle'
test_data_path = 'fraud_test.pickle'
encoder_path = 'fraud_encoder.hdf5'
decoder_path = 'fraud_decoder.hdf5'
autoencoder_path = 'fraud_autoencoder.hdf5'

# prepare input data
data = pd.read_csv(input_data_path).astype('float64')

# restrict data columns
#data = data.drop(columns = ['asn', 'longitude', 'latitude'])
#data = data.drop(columns = ['asn'])
print(data.shape)

# normalize data in selected columns
#data[['longitude', 'latitude']] = MinMaxScaler().fit_transform(data[['longitude', 'latitude']])
data[['asn', 'longitude', 'latitude']] = MinMaxScaler().fit_transform(data[['asn', 'longitude', 'latitude']])
print(data.head())

x_train, x_test = train_test_split(data, test_size=0.2)

# save train/test data
x_train.to_pickle(train_data_path)
x_test.to_pickle(test_data_path)

# this is the number of attributes
input_dim = data.shape[1]

# Single fully-connected neural layer as encoder and decoder

use_regularizer = False
my_regularizer = None
my_epochs = 200

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 400

# this is the size of our encoded representations
encoding_dim = 3   # 3 floats -> compression factor 12, assuming the input is 36 integers

# this is our input placeholder
input_x = Input(shape=(input_dim, ))

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_x)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_x, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_x, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# configure model to use a per-pixel binary crossentropy/MSQE loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train autoencoder for the preset number of epochs

autoencoder.fit(x_train, x_train, 
                epochs=my_epochs, 
                batch_size=1024, 
                shuffle=True, 
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# encode and decode some data
# note that we take them from the *test* set
encoded_x = encoder.predict(x_test)
decoded_x = decoder.predict(encoded_x)

# save model
encoder.save(encoder_path)
decoder.save(decoder_path)
autoencoder.save(autoencoder_path)

K.clear_session()
