import pandas as pd
from keras import backend as K
from keras import regularizers, optimizers
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model, load_model
from keras.backend import sigmoid
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ae_lib import Swish, iRprop_

input_data_path = "../data-full.csv"
train_data_path = 'fraud_old_train.pickle'
test_data_path = 'fraud_old_test.pickle'
encoder_path = 'fraud_encoder_deep_old.hdf5'
decoder_path = 'fraud_decoder_deep_old.hdf5'
autoencoder_path = 'fraud_autoencoder_deep_old.hdf5'
model_graph_path = 'fraud_autoencoder_deep_old_graph.png'

# prepare input data
data = pd.read_csv(input_data_path).astype('float64')

# restrict data columns
#data = data.drop(columns=['asn', 'longitude', 'latitude'])
#data = data.drop(columns=['asn'])
data = data.drop(columns=['is_chrome', 'is_firefox' ,'is_ie', 'is_opera' ,'is_other_browser', 'is_other_browser_v'])
#print(data.shape)

# normalize data in selected columns
#data[['longitude', 'latitude']] = MinMaxScaler().fit_transform(data[['longitude', 'latitude']])
data[['asn', 'longitude', 'latitude']] = MinMaxScaler().fit_transform(data[['asn', 'longitude', 'latitude']])
#print(data.head())

x_train, x_test = train_test_split(data, test_size=0.2)

# save train/test data
x_train.to_pickle(train_data_path)
x_test.to_pickle(test_data_path)

# this is the number of attributes
input_dim = data.shape[1]

# Single fully-connected neural layer as encoder and decoder

use_regularizer = False
my_regularizer = None
my_epochs = 500
my_batch = 1024
#my_batch = x_train.shape[0]

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 1000

# this is the size of the intermediate layer
intermediate_dim = 10

# this is the size of our encoded representations
encoding_dim = 3   # 3 floats -> compression factor 12, assuming the input is 36 integers

# this is our input placeholder
input_x = Input(shape=(input_dim, ))

# intermediate layer (encoder)
intermediate = Dense(intermediate_dim, activation='relu', activity_regularizer=my_regularizer)(input_x)

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(intermediate)
#encoded = Dense(encoding_dim, activity_regularizer=my_regularizer)(intermediate)
#encoded = LeakyReLU(alpha=0.05)(encoded)

# intermediate layer (decoder)
intermediate = Dense(intermediate_dim, activation='relu', activity_regularizer=my_regularizer)(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(intermediate)

# this model maps an input to its reconstruction
autoencoder = Model(input_x, decoded)
plot_model(autoencoder, to_file=model_graph_path, show_shapes=True, dpi=300)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_x, encoded)

# Separate Decoder model

# create a placeholder for an encoded (3-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last two layers of the autoencoder model
decoder_layer_1 = autoencoder.layers[-2]
decoder_layer_2 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer_2(decoder_layer_1(encoded_input)))

# configure model to use a per-attribute binary crossentropy/MSQE loss, and the Adadelta/iRprop_ optimizer
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer=iRprop_(), loss='binary_crossentropy')
#autoencoder.compile(optimizer=iRprop_(), loss='mean_squared_error')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10, monitor='val_loss')

# Train autoencoder for the preset number of epochs
autoencoder.fit(x_train, x_train, 
                epochs=my_epochs, 
                batch_size=my_batch, 
                shuffle=True, 
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), early_stopping_monitor])

# save model
encoder.save(encoder_path)
decoder.save(decoder_path)
autoencoder.save(autoencoder_path)

K.clear_session()
