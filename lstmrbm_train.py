from lstmrbm import *
from midi_parser import *

if __name__=='__main__':
    songs = get_songs('Test_Midi')
    print('{} songs processed'.format(len(songs)))
    
    # Hyperparameter
    n_hidden = 150
    n_hidden_recurrent = 100
    batch_size = 100
    n_epochs = 100

    model = LSTM_RBM(n_hidden, n_hidden_recurrent, batch_size, n_epochs)
    
    model.initialize_weights(songs)
    model.fit(songs)
    
    
