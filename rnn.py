from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import tensorflow as tf
import numpy as np

import random
import json
import pickle


def build_model(window, input_size):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(window, input_size)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len_chars))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model


def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    with np.errstate(divide='ignore'):
        preds = np.asarray(preds).astype('float64')

        preds = np.log(preds) / temperature        
        # Fix division by 0
        preds[preds == np.inf] = 0

        exp_preds = np.exp(preds)
        preds =  exp_preds / np.sum(exp_preds)
    
        return np.argmax(np.random.multinomial(1, preds, 1))
