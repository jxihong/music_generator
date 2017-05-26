from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import tensorflow as tf
import numpy as np

import random
import json
import pickle

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def build_model(window, len_notes):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(window, len_notes)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len_notes))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def sample(preds):
  # Helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  
  preds[preds == np.inf] = 0      
  return np.floor(preds + np.random.uniform(0, 1, preds.shape))
    
        
