import tensorflow as tf
import numpy as np

from rnnrbm import *
from midi_parser import *

# Extracts first timesteps as primer for generation
song_primer = 'Jazz_Music_Midi/PianoMan.mid'
# Saved weights for trainged rnnrbm
model_path = 'parameter_checkpoints/rnnrbm_final.ckpt'

if __name__=='__main__':
    num_songs = 5
    
    model = RNN_RBM()
    saver = tf.train.Saver(model.training_vars)
    
    start = get_song(midiToStatematrix(song_primer)) # Start sequence for generated song
    start_length = 600
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, model_path)
        
        for i in range(num_songs):
            music = sess.run(model.generate(300), 
                             feed_dict={ model.x: start[:2000], 
                                         model.music: start[:2000]})
            
            song_path = "generated/rnnrbm_{}.mid".format(i)
            write_song(song_path, music)

