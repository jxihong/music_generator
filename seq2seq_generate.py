import tensorflow as tf
import numpy as np

import random
import json
import pickle

from rnn import *

from midi_parser import *

import glob

from keras.models import model_from_json

WINDOW_SIZE = 10

note_range = span
n_visible = 2 * note_range * num_timesteps #The size of visible layer

song_primer = 'data/comrain_drums.mid'
song_melody = 'data/comrain_melody.mid'

if __name__=='__main__':
    window = 10

    json_file = open('parameter_checkpoints/rnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("parameter_checkpoints/seq2seq_epoch_50.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print "Loaded model from disk"
    
    start = get_song(midiToStatematrix(song_primer)) # Start sequence for generated song
    melody = get_song(midiToStatematrix(song_melody))

    start_length = 20
    for i in range(5):
        generated = start[:start_length]
        for j in range(300):
            x = np.expand_dims(melody[j:j + WINDOW_SIZE], axis = 0)
            
            preds = loaded_model.predict(x, verbose=0)[0]
            next = sample(preds)
            
            generated = np.vstack((generated, next))
            
        song_path = "generated/rnn_{}".format(i)
        write_song(song_path, generated)