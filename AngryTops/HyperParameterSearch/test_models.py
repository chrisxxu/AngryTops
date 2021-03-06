"""Models for hyperparmeter searches"""
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow import keras
import tensorflow as tf
from AngryTops.ModelTraining.custom_loss import *

metrics = ['mae', 'mse', weighted_MSE1, weighted_MSE2, w_HAD, w_LEP, b_HAD, b_LEP, t_HAD, t_LEP]

def test_model0(config):
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=True,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=True,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = LSTM(int(config['size3']), return_sequences=False,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act1'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act2'])(input_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size6']), activation=config['act3'])(combined)
    final = Dense(int(config['size7']), activation=config['act4'])(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model

def test_model1(config):
    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=False,
                  kernel_regularizer=l2(config['reg_weight']),
                  recurrent_regularizer=l2(config['rec_weight']))(x_jets)
    x_jets = Dense(int(config['size2']), activation=config['act1'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = keras.Model(inputs=input_lep, outputs=input_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size3']), activation=config['act2'])(combined)
    final = Dense(int(config['size4']), activation=config['act3'])(final)
    final = Dense(int(config['size5']), activation=config['act4'])(final)
    final = Dense(18, activation=config['act5'])(final)
    final = Reshape(target_shape=(6,3))(final)
    final = Dense(3, activation="linear")(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model

def test_model2(config):
    kernel_reg1 = config['kernel_reg1']
    kernel_reg2 = config['kernel_reg2']
    kernel_reg3 = config['kernel_reg3']
    kernel_reg4 = config['kernel_reg4']
    kernel_reg5 = config['kernel_reg5']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")
    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = LSTM(int(config['size1']), return_sequences=True, kernel_regularizer=l2(kernel_reg1))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=False, kernel_regularizer=l2(kernel_reg2))(x_jets)
    x_jets = Dense(int(config['size3']), activation=config['act1'], kernel_regularizer=l2(kernel_reg3))(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act2'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act3'])(input_lep)
    x_lep = Dense(int(config['size6']), activation=config['act4'])(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = BatchNormalization()(combined)
    final = Dense(int(config['size7']), activation=config['act5'], kernel_regularizer=l2(kernel_reg4))(final)
    final = Dense(int(config['size8']), activation=config['act6'], kernel_regularizer=l2(kernel_reg5))(final)
    final = Dense(18, activation='linear')(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model

def test_model3(config):
    kernel_reg1 = config['kernel_reg1']
    kernel_reg2 = config['kernel_reg2']
    kernel_reg3 = config['kernel_reg3']
    kernel_reg4 = config['kernel_reg4']
    kernel_reg5 = config['kernel_reg5']

    input_jets = Input(shape = (20,), name="input_jets")
    input_lep = Input(shape=(5,), name="input_lep")

    # Jets
    x_jets = Reshape(target_shape=(5,4))(input_jets)
    x_jets = Dense(int(config['size1']), activation=config['act1'], kernel_regularizer=l2(kernel_reg1))(x_jets)
    x_jets = LSTM(int(config['size2']), return_sequences=True, kernel_regularizer=l2(kernel_reg2))(x_jets)
    x_jets = LSTM(int(config['size3']), return_sequences=False, kernel_regularizer=l2(kernel_reg3))(x_jets)
    x_jets = BatchNormalization()(x_jets)
    x_jets = Dense(int(config['size4']), activation=config['act2'])(x_jets)
    x_jets = keras.Model(inputs=input_jets, outputs=x_jets)

    # Lep
    x_lep = Dense(int(config['size5']), activation=config['act3'], kernel_regularizer=l2(kernel_reg4))(input_lep)
    x_lep = BatchNormalization()(x_lep)
    x_lep = Dense(int(config['size6']), activation=config['act4'])(x_lep)
    x_lep = keras.Model(inputs=input_lep, outputs=x_lep)

    # Combine them
    combined = concatenate([x_lep.output, x_jets.output], axis=1)

    # Apply some more layers to combined data set
    final = Dense(int(config['size7']), activation=config['act5'], kernel_regularizer=l2(kernel_reg5))(combined)
    final = Dense(18)(final)
    final = Reshape(target_shape=(6,3))(final)

    # Make final model
    model = keras.Model(inputs=[x_lep.input, x_jets.input], outputs=final)

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model

def test_model4(config):
    """A denser version of model_multi"""
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    model.add(TimeDistributed(Dense(int(config['size1']), activation='tanh')))
    model.add(LSTM(int(config['size2']), return_sequences=True))
    #model.add(TimeDistributed(Dense(108, activation='tanh')))
    model.add(LSTM(int(config['size3']), return_sequences=True))
    #model.add(TimeDistributed(Dense(72, activation='tanh')))
    model.add(LSTM(int(config['size4']), return_sequences=True))
    #model.add(TimeDistributed(Dense(36, activation='tanh')))
    model.add(LSTM(int(config['size5']), return_sequences=True))
    #model.add(TimeDistributed(Dense(18, activation='tanh')))
    model.add(LSTM(3, return_sequences=True))
    #model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-4, decay=0.)
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model

def test_model5(config):
    """A denser version of model_multi"""
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    # Initially, due to typo, size1 = size2
    model.add(TimeDistributed(Dense(int(config['size1']), activation=config['act1'])))
    model.add(Bidirectional(LSTM(int(config['size2']), return_sequences=True)))
    model.add(TimeDistributed(Dense(int(config['size3']), activation=config['act2'])))
    model.add(TimeDistributed(Dense(int(config['size4']), activation=config['act3'])))
    model.add(TimeDistributed(Dense(int(config['size5']), activation=config['act4'])))
    model.add(TimeDistributed(Dense(3, activation='linear')))

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.)
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    return model


def cnn_test1(config):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,), activation='relu'))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(int(config['size1']), activation=config['act1']))
    model.add(Dense(int(config['size2']), activation=config['act2']))
    model.add(Dense(int(config['size3']), activation=config['act3']))
    model.add(Dense(18))
    model.add(Reshape(target_shape=(6,3)))

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
    return model

def cnn_test2(config):
    """A simple convolutional network model"""
    model = keras.models.Sequential()
    model.add(Dense(256, input_shape=(36,), activation='relu'))
    model.add(Reshape(target_shape=(8,8,4)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(int(config['size1']), activation=config['act1']))
    model.add(Dense(int(config['size2']), activation=config['act2']))
    model.add(Dense(int(config['size3']), activation=config['act3']))
    model.add(Dense(int(config['size4']), activation=config['act4']))
    model.add(Dense(18))
    model.add(Reshape(target_shape=(6,3)))

    optimizer = tf.keras.optimizers.Adam(config['learn_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
    return model

test_models = {'test_model0': test_model0, 'test_model1': test_model1,
               'test_model2': test_model2, 'test_model3': test_model3, 'test_model4': test_model4,
               'cnn_test1': cnn_test1, 'cnn_test2': cnn_test2,
               'test_model5':test_model5}
