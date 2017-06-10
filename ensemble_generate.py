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


if __name__=='__main__':
    json_file = open('parameter_checkpoints/rnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    bass_model = model_from_json(loaded_model_json)
    drum_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    bass_model.load_weights("parameter_checkpoints/seq2seq_bass_final.h5")
    bass_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    drum_model.load_weights("parameter_checkpoints/seq2seq_drum_final.h5")
    drum_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print "Loaded model from disk"
    

    start_length = 20
    for i in range(5):
        song_melodies = glob.glob('generated/rnnrbm*mid')
        song_melody = np.random.choice(song_melodies)
        print song_melody

        melody = get_song(midiToStatematrix(str(song_melody)))

        drum_generated = np.zeros(2 * note_range)
        bass_generated = np.zeros(2 * note_range)
        
        onNotes = np.zeros(note_range) # stores duration each note is currently being played
        for j in range(1000):
            x = np.expand_dims(melody[j:j + WINDOW_SIZE], axis = 0)
            
            drum_preds = drum_model.predict(x, verbose=0)[0]
            drum_next = sample(drum_preds)
            
            bass_preds = bass_model.predict(x, verbose=0)[0]
            
            bass_next = sample(bass_preds)
            # Cleaning function for bass
            if j > 0:
                if np.sum(bass_next[:note_range]) > 4:
                    bass_next = np.concatenate((bass_generated[-1][:note_range],[0] * note_range))
            
                for k in range(note_range):
                    if (bass_next[k] == 0):
                        if (onNotes[k] <= 4):
                            bass_generated[-1][k] = 0
                            onNotes[k] == 0
                        else:
                            onNotes[k] += 1

            drum_generated = np.vstack((drum_generated, drum_next))
            bass_generated = np.vstack((bass_generated, bass_next))
            
        
        generated_drum_pattern = statematrixToPattern(drum_generated)
        generated_bass_pattern = statematrixToPattern(bass_generated)
        melody_pattern = statematrixToPattern(melody)
        
        for track in generated_drum_pattern:
            melody_pattern.append(track)
        for track in generated_bass_pattern:
            melody_pattern.append(track)
            
        song_path = "generated/emsemble_rnn_{}".format(i)

        midi.write_midifile("{}.mid".format(song_path), melody_pattern)
        
        
