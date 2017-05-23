from lstmrbm import *
from midi_parser import *

if __name__=='__main__':
    #songs = get_songs('Test_Midi')
    
    songs = []
    for file in glob.glob("Classical_Data/*txt"):
        song = np.genfromtxt(file)
        songs.append(song)
        
    '''
    songs = []
    for file in glob.glob("Jazz_Data/*_melody.txt"):
        song = np.genfromtxt(file)
        try:
            song = song[:int(np.floor((song.shape[0]/num_timesteps) * num_timesteps))]
            # Reshape into blocks of num_timesteps
            song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
            if song.shape[0] > 50:
                songs.append(song)
        except:
            continue
    '''
    
    print('{} songs processed'.format(len(songs)))
    
    # Hyperparameter
    n_hidden = 150
    n_hidden_recurrent = 100
    batch_size = 100
    n_epochs = 400

    model = LSTM_RBM(n_hidden, n_hidden_recurrent, batch_size, n_epochs)
    
    model.initialize_weights(songs)
    model.fit(songs)
    
    
