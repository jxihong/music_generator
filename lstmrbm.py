from tqdm import tqdm
import numpy as np
import tensorflow as tf

from rbm import * 
from tensorflow.contrib import rnn

import midi_parser

note_range         = midi_parser.span
n_visible          = 2 * note_range * midi_parser.num_timesteps #The size RBM visible layer

# Variable names and general guideline for model are taken from deeplearning.net tutorial.
# Tensorflow code was semi-adapted from the site's Theanos code. 
# Source: http://deeplearning.net/tutorial/rnnrbm.html

def build_lstmrbm(n_hidden, n_hidden_recurrent):
    """
    Builds a LSTM-RBM model.
    
    n_hidden: Number of hidden units in RBM
    n_hidden_recurrent: Number of hidden units of RNN
    """
    x  = tf.placeholder(tf.float32, [None, n_visible]) #input data
    lr  = tf.placeholder(tf.float32) #The learning rate

    batch_size = tf.shape(x)[0] 
    
    # Initialize parameters of model
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")    
    bu = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")

    # RBM parameters
    W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")
    
    # LSTM parameters
    Wui = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wui")
    Wqi = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wqi")
    Wci = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wci")
    bi = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bi")
    
    Wuf = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuf")
    Wqf = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wqf")
    Wcf = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wcf")
    bf = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bf")
    
    Wuc = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuc")
    Wqc = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wqc")
    bc = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bc")
    
    Wuo = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuo")
    Wqo = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wqo")
    Wco = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wco")
    Wqv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wqv")
    Wqh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wqh")
    bo = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bo")

    params =  [W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, Wui, Wqi, Wci, bi, 
               Wuf, Wqf, Wcf, bf, Wuc, Wqc, bc, Wuo, Wqo, Wco, bo, Wqv, Wqh]
    
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    q0 =  tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="q0")
    c0 = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="c0")

    batch_bh_t = tf.Variable(tf.zeros([1, n_hidden]), name="batch_bh_t")
    batch_bv_t = tf.Variable(tf.zeros([1, n_visible]), name="batch_bv_t")
    
    music = tf.placeholder(tf.float32, [None, n_visible])
    
    def lstm_recurrence(lstm_args, v_t):
        u_tm1 = lstm_args[0]
        q_tm1 = lstm_args[1]
        c_tm1 = lstm_args[2]
        
        v_t = tf.reshape(v_t, [1, n_visible])
        
        u_t  = (tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu)))
        
        i_t = tf.tanh(bi + tf.matmul(c_tm1, Wci) + tf.matmul(q_tm1, Wqi) + tf.matmul(u_t, Wui))
        f_t = tf.tanh(bf + tf.matmul(c_tm1, Wcf) + tf.matmul(q_tm1, Wqf) + tf.matmul(u_t, Wuf))
        c_t = (f_t * c_tm1) + (i_t * tf.tanh(tf.matmul(u_t, Wuc) + tf.matmul(q_tm1, Wqc) + bc))
        o_t = tf.tanh(bo + tf.matmul(c_t, Wco) + tf.matmul(q_tm1, Wqo) + tf.matmul(u_t, Wuo))
        q_t = o_t * tf.tanh(c_t)
        return u_t, q_t, c_t
    

    def bv_recurrence(bv_t, lstm_args):
        u_tm1 = lstm_args[0]
        q_tm1 = lstm_args[1]
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv)) + tf.matmul(q_tm1, Wqv)
        return bv_t


    def bh_recurrence(bh_t, lstm_args):
        u_tm1 = lstm_args[0]
        q_tm1 = lstm_args[1]
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh)) + tf.matmul(q_tm1, Wqh)
        return bh_t       

        
    def recurrence(i, k, v_tm1, u_tm1, q_tm1, c_tm1, music):
        #Get the bias vectors from the current state of the RNN
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv)) + tf.matmul(q_tm1, Wqv)
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh)) + tf.matmul(q_tm1, Wqh)
        
        if v_tm1 is None:
            v_t = gibbs_sample(tf.zeros([1, n_visible], tf.float32), W, bv_t, bh_t, k=25)
        else:
            v_t = gibbs_sample(v_tm1, W, bv_t, bh_t, k=25)
            
        u_t  = (tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu)))
        
        i_t = tf.tanh(bi + tf.matmul(c_tm1, Wci) + tf.matmul(q_tm1, Wqi) + tf.matmul(u_t, Wui))
        f_t = tf.tanh(bf + tf.matmul(c_tm1, Wcf) + tf.matmul(q_tm1, Wqf) + tf.matmul(u_t, Wuf))
        c_t = (f_t * c_tm1) + (i_t * tf.tanh(tf.matmul(u_t, Wuc) + tf.matmul(q_tm1, Wqc) + bc))
        o_t = tf.tanh(bo + tf.matmul(c_t, Wco) + tf.matmul(q_tm1, Wqo) + tf.matmul(u_t, Wuo))
        q_t = o_t * tf.tanh(c_t)
        
        music = tf.concat(0, [music, v_t])
        return [i + 1, k, v_t, u_t, q_t, c_t, music]
    
    
    def generate_music(num_timesteps, x=x, music_init=music, n_visible=n_visible,
                       lstm_args=(u0, q0, c0), start_length=200):
        """
        Generates sequence of music.
        """
        batch_u, batch_q, batch_c = tf.scan(lstm_recurrence, x, initializer=lstm_args)
        # Round down and get starting sequence
        u_init = batch_u[int(np.floor(start_length/midi_parser.num_timesteps)), :, :]
        q_init = batch_q[int(np.floor(start_length/midi_parser.num_timesteps)), :, :]
        c_init = batch_c[int(np.floor(start_length/midi_parser.num_timesteps)), :, :]
        
        i = tf.constant(1, tf.int32)
        k = tf.constant(num_timesteps)
        v_init = tf.zeros([1, n_visible])
                
        [_, _, _, _, _, _, music] = tf.while_loop(lambda i, k, *args: i < k, 
                                                  recurrence, 
                                                  [i, k, v_init, u_init, q_init, c_init, music_init])
        
        return music
        
    
    # Reshape biases to match batch_size
    tf.assign(batch_bh_t, tf.tile(batch_bh_t, [batch_size, 1]))
    tf.assign(batch_bv_t, tf.tile(batch_bv_t, [batch_size, 1]))
    
    # For training, compute bv_t, bh_t given x. 
    batch_u_t, batch_q_t, batch_c_t = tf.scan(lstm_recurrence, x, initializer=(u0, q0, c0))
    
    batch_bh_t =  tf.reshape(tf.scan(bh_recurrence, [batch_u_t, batch_q_t], 
                                     tf.zeros([1, n_hidden], tf.float32)),
                             [batch_size, n_hidden])

    batch_bv_t =  tf.reshape(tf.scan(bv_recurrence, [batch_u_t, batch_q_t], 
                                     tf.zeros([1, n_visible], tf.float32)),
                             [batch_size, n_visible])
    
    
    # Get free energy cost
    cost = get_free_energy_cost(x, W, batch_bv_t, batch_bh_t, k=15)
    
    return x, cost, generate_music, params, u0, q0, c0, lr, music


class LSTM_RBM:
    """
    Class to train an RNN-RBM from MIDI and generate music.
    """
    def __init__(self, n_hidden=150, n_hidden_recurrent=100, 
                 batch_size=100, n_epochs=500):
        """
        Constructs a RNN-RBM with training and sequence generation functions.
        """
        self.x, cost, generate, params, u0, q0, c0, \
            self.learning_rate, self.music = build_lstmrbm(n_hidden, n_hidden_recurrent)

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Variables for training
        self.training_vars = params + [u0, q0, c0] 
        self.cost = cost
        
        opt_func = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        gradients = opt_func.compute_gradients(cost, self.training_vars)
        
        self.update = opt_func.apply_gradients(gradients) # Update step
        
        self.generate = generate
        
        
    def initialize_weights(self, songs, save="parameter_checkpoints/lstmrbm_initial.ckpt"):
        """
        Initialize the RBM weights from Contrastive Divergence
        """
        W, bv, bh = self.training_vars[:3]
        rbm_update = cd_update(self.x, W, bv, bh, self.learning_rate)
                
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(self.n_epochs):
                for song in songs:
                    for i in range(1, len(song), self.batch_size):
                        batch = song[i: i + self.batch_size]
                        sess.run(rbm_update, feed_dict={self.x: batch,
                                                        self.learning_rate:5e-3})
                        
            save_path = saver.save(sess, save)


    def fit(self, songs, 
            checkpoint="parameter_checkpoints/lstmrbm_initial.ckpt", 
            save="parameter_checkpoints/lstmrbm_final.ckpt"):
        """
        Train RNN-RBM via SGD on parsed MIDI matrices.
        """
        saver = tf.train.Saver(self.training_vars)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            # Load initial weights of model
            if len(checkpoint) > 1:
                saver.restore(sess, checkpoint)
                
            for epoch in range(self.n_epochs):
                costs = []
                for song in songs:
                    for i in range(1, len(song), self.batch_size):
                        batch = song[i: i + self.batch_size]
                        # Adaptive learning rate
                        alpha = min(0.01, 0.1/float(i))
                        _, cost = sess.run([self.update, self.cost], feed_dict={self.x:batch, 
                                                                                self.learning_rate:alpha})
                        costs.append(cost)
                print("Epoch: {}/{}. Cost: {}".format(epoch, self.n_epochs, np.mean(costs)))
                
                if (epoch + 1) % 50 == 0:
                    saver.save(sess, "parameter_checkpoints/lstmrbm_epoch_{}.ckpt".format(epoch + 1))
                    
            saver.save(sess, save)

