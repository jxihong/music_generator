from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils import *
from rbm import * 

import midi_parser

note_range         = midi_parser.span
n_visible          = 2 * note_range * midi_parser.num_timesteps #The size RBM visible layer

# Variable names and general guideline for model are taken from deeplearning.net tutorial.
# Tensorflow code was semi-adapted from the site's Theanos code. 
# Source: http://deeplearning.net/tutorial/rnnrbm.html

def build_rnnrbm(n_hidden, n_hidden_recurrent):
    """
    Builds a RNN-RBM model.
    
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

    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    
    def u_recurrence(u_tm1, v_t):
        u_t = (tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tmq, Wuu)))
        return u_t
    
    def bv_recurrence(bv_t, u_tm1):
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        return bv_t

    def bh_recurrence(bh_t, u_tm1):
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t       
        
    def recurrence(i, k, v_tm1, u_tm1, music):
        #Get the bias vectors from the current state of the RNN
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        
        if v_tm1 is None:
            v_t = rbm.gibbs_sample_converge(tf.zeros([1, n_visible], tf.float32), W, bv_t, bh_t)
        else:
            v_t = rbm.gibbs_sample_converge(v_tm1, W, bv_t, bh_t)
        
        u_t  = (tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu)))
        
        music = tf.concat(0, [music, v_t])
        return i + 1, k, v_t, u_t, music
    
    
    def generate_music(num_timesteps, x=x, n_visible=n_visible,
                       u0=u0, start_length=100):
        """
        Generates sequence of music.
        """
        batch_u = tf.scan(u_recurrence, x, initializer=u0)
        # Round down and get starting sequence
        batch_u = batch_u[np.floor(start_length/midi_parser.num_timesteps), :, :]
        
        [_, _, _, music] = tf.while_loop(lambda i, k, *args: i < k, 
                                         recurrence, 
                                         [tf.constant(1, tf.int32), 
                                          tf.constant(num_timesteps),
                                          x,
                                          tf.zeros([1, n_visible], tf.float32),
                                          tf.zeros([1, n_visible], tf.float32)])
        return music
        
    
    
    batch_bh_t = tf.Variable(tf.zeros([batch_size, n_hidden]), name="batch_bh_t")
    batch_bv_t = tf.Variable(tf.zeros([batch_bv_t, n_visible]), name="batch_bv_t")

    # For training, compute bv_t, bh_t given x. 
    batch_u_t = tf.scan(u_recurrence, x, initializer=u0)
    
    batch_bh_t =  tf.reshape(tf.scan(bh_recurrence, batch_u_t, tf.zeros([1, n_hidden], tf.float32)), 
                             [batch_size, n_hidden])
    batch_bv_t =  tf.reshape(tf.scan(bv_recurrence, batch_u_t, tf.zeros([1, n_hidden], tf.float32)), 
                             [batch_size, n_hidden])
    
    # Get free energy cost
    cost = rbm.get_free_energy_cost(x, W, batch_bv_t, batch_bh_t, k=15)
    
    return x, cost, generate_music, W, bh, bv, lr, Wuh, Wuv, Wvu, Wuu, bu, u0


class RNN_RBM:
    """
    Class to train an RNN-RBM from MIDI and generate music.
    """
    pass
