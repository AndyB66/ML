import pandas as pd
from keras import backend as K
from keras import regularizers, optimizers
from keras.layers import Input, Dense
from keras.layers import LeakyReLU, Activation
from keras.models import Model
from keras.models import load_model
from keras.backend import sigmoid
from keras.optimizers import Optimizer, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x, beta = 1):
    return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'swish': Swish(swish)})

class iRprop_(Optimizer):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(iRprop_, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)    
            )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha 

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

input_data_path = "data-full.csv"
train_data_path = 'fraud_train.pickle'
test_data_path = 'fraud_test.pickle'
encoder_path = 'fraud_encoder.hdf5'
decoder_path = 'fraud_decoder.hdf5'
autoencoder_path = 'fraud_autoencoder.hdf5'
model_graph_path = 'fraud_autoencoder_graph.png'

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

# this is the size of our encoded representations
encoding_dim = 3   # 3 floats -> compression factor 12, assuming the input is 36 integers

# this is our input placeholder
input_x = Input(shape=(input_dim, ))

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='swish', activity_regularizer=my_regularizer)(input_x)
#encoded = Dense(encoding_dim, activity_regularizer=my_regularizer)(input_x)
#encoded = LeakyReLU(alpha=0.05)(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_x, decoded)
plot_model(autoencoder, to_file=model_graph_path, show_shapes=True, dpi=300)

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
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer=iRprop_(), loss='binary_crossentropy')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10, monitor='val_loss')

# Train autoencoder for the preset number of epochs
autoencoder.fit(x_train, x_train, 
                epochs=my_epochs, 
                batch_size=my_batch, 
                shuffle=True, 
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), early_stopping_monitor])

# encode and decode some data
# note that we take them from the *test* set
encoded_x = encoder.predict(x_test)
decoded_x = decoder.predict(encoded_x)

# save model
encoder.save(encoder_path)
decoder.save(decoder_path)
autoencoder.save(autoencoder_path)

K.clear_session()
