import tensorflow as tf
import numpy as np

from lstmrbm import *
from midi_parser import *

# Extracts first timesteps as primer for generation
song_primer = 'Classical_Music_Midi/C_jigs_simple_chords_129.mid'
# Saved weights for trainged rnnrbm
model_path = 'parameter_checkpoints/lstmrbm_epoch_200.ckpt'

if __name__=='__main__':
    num_songs = 5
    
    model = LSTM_RBM()
    saver = tf.train.Saver(model.training_vars)
    
    start = get_song(midiToStatematrix(song_primer)) # Start sequence for generated song
    start_length = 20
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, model_path)
        
        for i in range(num_songs):
            music = sess.run(model.generate(300), 
                             feed_dict={ model.x: start[:201], 
                                         model.music: start[:start_length]})
            
            song_path = "generated/lstmrbm_{}".format(i)
            write_song(song_path, music)

