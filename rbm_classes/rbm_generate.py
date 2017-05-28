from RBM import *
from midi_parser import *

import sys
sys.path.append('../')

import glob

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in files:
        try:
            song = np.array(midiToStatematrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except:
            # Just ignore songs that can't be parsed
            continue         
    return songs


if __name__=='__main__':
    songs = get_songs('../Jazz_Music_Midi')
    print "{} songs processed".format(len(songs))
    
    ### HyperParameters

    lowest_note = 24
    highest_note = 102 
    note_range = highest_note-lowest_note

    num_timesteps = 5
    # Size of input layer
    input_len = 2 * note_range * num_timesteps
    
    model = RBM(n_epochs=200)
    
    X = []
    for song in songs:
        song = np.array(song)
        # Round down to nearest multiple
        song = song[:int(np.floor((song.shape[0]/num_timesteps) * num_timesteps))]
        # Reshape into blocks of num_timesteps
        song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
        X.extend(song)
    X = np.array(X)
    
    model.fit(X)
    
    gen = model.sample(0.001 * np.ones((10, input_len)))
    
    for i in range(gen.shape[0]):
        if not any(gen[i, :]):
            continue
        #Here we reshape the vector to be time x notes, and then save the vector as a midi file
        s = np.reshape(gen[i,:], (num_timesteps, 2*note_range))
        statematrixToMidi(s, "generated_{}".format(i))
