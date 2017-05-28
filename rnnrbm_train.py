from rnnrbm import *
from midi_parser import *

num_timesteps = midi_parser.num_timesteps

if __name__=='__main__':
    # There are two different ways to get song data.
    # (1) Directly from MIDI files
    # (2) From statematrices that are not reshaped yet
    #
    # Uncomment the method you use

    
    jazz_songs = get_songs('Jazz_Music_Midi')
    
    songs = []
    for file in glob.glob("Classical_Data/*chords*.txt"):
        song = np.genfromtxt(file)
        try:
            song = song[:int(np.floor((song.shape[0]/num_timesteps) * num_timesteps))]
            # Reshape into blocks of num_timesteps
            song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
            if song.shape[0] > 50/num_timesteps:
                songs.append(song)
        except:
            pass
    print('{} songs processed'.format(len(songs)))
    
    # Hyperparameter
    n_hidden = 150
    n_hidden_recurrent = 100
    batch_size = 100
    n_epochs = 200
    
    model = RNN_RBM(n_hidden, n_hidden_recurrent, batch_size, n_epochs)
    
    model.initialize_weights1(songs)
    model.fit(songs)
    #model.fine_tune_weights(jazz_songs)
    
