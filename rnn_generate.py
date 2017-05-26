import tensorflow as tf
import numpy as np

import random
import json
import pickle

from rnn import *

from midi_parser import *

import glob

from keras.models import model_from_json

note_range = span
n_visible = 2 * note_range * num_timesteps #The size of visible layer

song_primer = 'Test_Midi/PianoMan.mid'

if __name__=='__main__':
    window = 10

    #json_file = open('parameter_checkpoints/rnn.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    
    loaded_model = build_model(window, n_visible)
    
    # load weights into new model
    loaded_model.load_weights("parameter_checkpoints/rnn_final.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print "Loaded model from disk"
    
    start = get_song(midiToStatematrix(song_primer)) # Start sequence for generated song

    start_length = 20
    for i in range(5):
        generated = start[:start_length]
        while len(generated) < 300:
            x = np.expand_dims(generated[-10:], axis = 0)
            
            preds = loaded_model.predict(x, verbose=0)[0]
            next = sample(preds)
            
            generated = np.vstack((generated, next))
            
        song_path = "generated/rnn_{}".format(i)
        write_song(song_path, generated)
        
    
        
        
    
