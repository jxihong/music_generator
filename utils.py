"""
Contains various data utilities. 
"""

import glob
import numpy as np
# from midi_parser import midiToStatematrix
from rnn import *

NOTE_SIZE = 156

def batch_generator(data, batch_size):
    """
    Generates batches of samples of size batch_size
    """
    data = np.array(data)
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        batch = data_shuffled[start:end]
        if len(batch) < batch_size:
            # Pad with zeros                                                               
            pad = np.zeros((batch_size - batch.shape[0], batch.shape[1]),
                           dtype=batch.dtype)
            batch = np.vstack((batch, pad))

        yield batch


def write_state_matrices():
    """
    Writes the state matrices of all the melody and drum files to a folder 
    called matrices in the data directory. 
    """
    drum_files = glob.glob('./data/*_drums.mid')
    melody_files = glob.glob('./data/*_melody.mid')
    for i in range(len(drum_files)):
        drum_fname = "./data/matrices/" + drum_files[i].split("/")[-1][:-4] + ".txt"
        melody_fname = "./data/matrices/" + melody_files[i].split("/")[-1][:-4] + ".txt"
        drum_mat = midiToStatematrix(drum_files[i])
        melody_mat = midiToStatematrix(melody_files[i])

        # Pad the matrices to have the same length.

        len_drum = len(drum_mat)
        len_melody = len(melody_mat)
        min_mat = melody_mat
        if(len_drum < len_melody):
            min_mat = drum_mat
        diff = abs(len_drum - len_melody)
        zeros = np.zeros((diff,NOTE_SIZE))
        min_mat = np.concatenate((min_mat,zeros))
        if(len_drum < len_melody):
            drum_mat = min_mat
        else:
            melody_mat = min_mat

        # Write the two matrices
        np.savetxt(drum_fname,drum_mat)
        np.savetxt(melody_fname,melody_mat) 


def read_state_matrices():
    """
    Reads in the written state matrices from the files, returns two arrays
    of state matrices. 
    """
    drum_mats = glob.glob('./Melody_Bass_Data/*_bass.txt')
    melody_mats = glob.glob('./Melody_Bass_Data/*_melody.txt')
    
    drum_states = []
    melody_states = []

    for drum_mat,melody_mat in zip(drum_mats[:20],melody_mats[:20]):
        temp= np.genfromtxt(drum_mat)
        drum_states.append(temp)
        temp= np.genfromtxt(melody_mat)	
        melody_states.append(temp)
        
    return drum_states,melody_states


def pad_track_list(stateList):   
    '''
    This function takes a list of tracks which are themselves numpy arrays 
    of (1,156) numpy arrays which represent notes. It pads the end of each track 
    with enough notevectors of all zeros so that every track has the length of 
    the longest initial track. It returns an updated tracklist.
    '''
    trackList = stateList
    lengths = []
    for i in range(0, len(trackList)):
        lengths.append(len(trackList[i]))
    
    longest_track = lengths.index(max(lengths))
    max_length = max(lengths)
    
    for i in range(0, len(trackList)):
        zeros = np.zeros((max_length - len(trackList[i]), 156))
        trackList[i] = np.concatenate((trackList[i], [zeros]))
    return trackList

def import_seq_data():
    '''
    This function returns the padded data in X,Y form, with X being the padded
    melody matrices, and Y being the padded drum matrices.
    '''
    drums, melodies = read_state_matrices()
    
    return melodies,drums
