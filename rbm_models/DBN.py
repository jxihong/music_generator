from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils import *
from rbm import *

class DBN():
    """
    Implements an unsupervised Deep Belief Network.
    """
    
    def __init__(self, 
                 rbm_n_hidden=[100, 100], 
                 rbm_learning_rate = 5e-3,
                 rbm_batch_size = 100, 
                 rbm_n_epochs=500,
                 session = tf.Session(),
                 batch_size = 100,
                 n_epochs=500,
                 model_path="models/dbn.ckpt"):
        
        self.sess = session
        
        self.rbm_layers = list()
        for n_hidden in rbm_n_hidden:
            rbm = RBM(n_hidden=n_hidden, 
                      learning_rate=rbm_learning_rate, 
                      batch_size=rbm_batch_size, 
                      n_epochs=rbm_n_epochs,
                      session=self.sess)
            
            self.rbm_layers.append(rbm)
            
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_path = model_path
        
        
    def __del__(self):
        self.sess.close()
        
        
    def fit(self, X):
        """
        Fit a model given data. 
        X: shape = (n_samples, n_features)
        """
        self.n_visible = X.shape[1]
        
        input = X
        # Greedily train each RBM in the DBN
        for rbm in self.rbm_layers:
            rbm.fit(input)
            input = rbm.compute_hidden(input)
            
        self.fine_tune(X)

    
    def fine_tune(self, X):
        """
        Fine-tune weights with contrastive version of 'wake-sleep' algorithm.
        """
        for batch in batch_generator(X, self.batch_size):
            for rbm in self.rbm_layers[:-1]:
                rbm.transform(batch)
                batch = rbm.compute_hidden(batch)
            
                top = self.rbm_layers[-1]
                top.fit(batch)
                
                for rbm in self.rbm_layers[-2::-1]:
                    rbm.transform_down(batch)
                    batch = rbm.compute_visible(batch)
                
                
    def sample(self, input):
        """
        Sample given visible units using Top-down sampling.
        """
        sample = self.gibbs_sample().eval(session=self.sess,
                                          feed_dict={self.x: input})
        
        return sample

    
    def gibbs_sample(self):
        """
        Run Gibbs Sampling on multiple layers.
        """
        self.x = tf.placeholder(tf.float32, [None, self.n_visible], name="x")
        
        h = self.x
        for rbm in self.rbm_layers[:-1]:
            h = sample(tf.sigmoid(tf.matmul(h, rbm.W) + rbm.hb))
            
        top = self.rbm_layers[-1]
        top.x = h
        self.x = top.gibbs_sample_converge()
            
        for rbm in self.rbm_layers[-2::-1]:
            self.x = sample(tf.sigmoid(tf.matmul(self.x, tf.transpose(rbm.W)) + rbm.vb))
         
        # TF tutorials said we need this to stop RBM values from backpropogating
        x_sample = tf.stop_gradient(self.x) 
        return x_sample
    
