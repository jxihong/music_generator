import tensorflow as tf
import numpy as np

import random
import json
import pickle

from rnn import *

import midi_parser
import glob

note_range         = midi_parser.span
n_visible          = 2 * note_range * midi_parser.num_timesteps #The size of visible layer

if __name__=='__main__':
    # Extract songs
    songs = []
    for file in glob.glob("Classical_Data/*txt")[:50]:
        song = np.genfromtxt(file)
        if song.shape[0] > 50:
            songs.append(song)
        
    # Hyperparameters
    window = 10
    input_size = n_visible
    
    # Build training data
    X = []
    y = []
    for song in songs:
        for i in range(1, len(song), window):
            if (i + window >= len(song)):
                break
            X.append(song[i: i + window])
            y.append(song[i + window])
            
    model = build_model(window, n_visible)
    
    X = np.array(X)
    y = np.array(y)
    
    # serialize model
    model_json = model.to_json()
    with open("parameter_checkpoints/rnn.json", "w") as json_file:
        json_file.write(model_json)

    for i in range(4):
        model.fit(X, y, batch_size=100, nb_epoch=50)
        
        print("Saved checkpoint after {} epochs.".format(i * 50))
        # serialize weights to HDF5
        model.save_weights("parameter_checkpoints/rnn_epoch_{}.h5".format(i * 50))
            
    # serialize weights to HDF5
    model.save_weights("parameter_checkpoints/rnn_final.h5")
    
