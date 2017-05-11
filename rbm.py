import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils import *

def sample(probs):
    """ 
    Takes in a vector of probabilities, and returns a random vector 
    of 0s and 1s sampled from the input vector
    """
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


class RBM():
    """
    Implements a Restricted Boltzmann machine. Follows Sci-kit learn conventions,
    but fairly loosely.
    """
    
    def __init__(self, n_hidden=100, learning_rate = 5e-3,
                 batch_size = 100, n_epochs=500,
                 model_path="models/rbm.ckpt"):

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
        self.model_path = model_path

        
    def fit(self, X):
        """
        Fit a model given data. 
        X: shape = (n_samples, n_features)
        """
        self.n_visible = X.shape[1]
        
        self.W  = tf.Variable(tf.random_normal([self.n_visible, self.n_hidden], 0.01), name="W") 
        # bias hidden layer
        self.hb = tf.Variable(tf.zeros([1, self.n_hidden],  tf.float32, name="hb"))
        # bias visible layer
        self.vb = tf.Variable(tf.zeros([1, self.n_visible],  tf.float32, name="vb"))
        
        # Get training process
        self._train_model()
        
        saver = tf.train.Saver()
        # Train the model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            #Run through all of the training data num_epochs times
            for epoch in tqdm(range(self.n_epochs)):
                for batch in batch_generator(X, self.batch_size):
                    sess.run(self.update, feed_dict={self.x:batch})
            save_path = saver.save(sess, self.model_path)
            print("Model saved in file: %s" %save_path)
            
    def sample(self, input):
        """
        Sample from distribution
        """
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            sample = self.gibbs_sample(10).eval(feed_dict={self.x: input})
        
        return sample
    

    def gibbs_sample(self, k): # iterates for k steps
        """
        Run Gibbs Sampling for k iterations 
        """
        self.x = tf.placeholder(tf.float32, [None, self.n_visible], name="x") 

        def step(i, k, x_k):
            #Propagate the visible values to sample the hidden values
            h_k = sample(tf.sigmoid(tf.matmul(x_k, self.W) + self.hb)) 
            #Propagate the hidden values to sample the visible values
            x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(self.W)) + self.vb)) 
            return i + 1, k, x_k
    
        i = tf.constant(0) #counter
        [_, _, x_sample] = tf.while_loop(lambda i, k, *args: i < k, step,
                                         [i, tf.constant(k), self.x], 
                                         parallel_iterations=1, 
                                         back_prop=False)
    
        # TF tutorials said we need this to stop RBM values from backpropogating
        x_sample = tf.stop_gradient(x_sample) 
        return x_sample
    
    
    def _train_model(self):
        """
        Run contrastive divergence to get weight updates for a single
        epoch of training.
        """
        self.x  = tf.placeholder(tf.float32, [None, self.n_visible], name="x") 

        # Run gibbs sampling for one step and save samples for x and h
        h = sample(tf.sigmoid(tf.matmul(self.x, self.W) + self.hb)) 
        
        x_sample = sample(tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vb)) 
        h_sample = sample(tf.sigmoid(tf.matmul(x_sample, self.W) + self.hb)) 

        #Update the values of W, hb, and vb
        size_x = tf.cast(tf.shape(self.x)[0], tf.float32)

        W_update  = tf.mul(self.learning_rate/size_x, 
                           tf.sub(tf.matmul(tf.transpose(self.x), h), \
                                      tf.matmul(tf.transpose(x_sample), h_sample)))
        vb_update = tf.mul(self.learning_rate/size_x, 
                           tf.reduce_sum(tf.sub(self.x, x_sample), 0, True))
        hb_update = tf.mul(self.learning_rate/size_x, 
                           tf.reduce_sum(tf.sub(h, h_sample), 0, True))

        #When we do sess.run(updt), TensorFlow will run all 3 update steps
        self.update = [self.W.assign_add(W_update), self.vb.assign_add(vb_update), 
                       self.hb.assign_add(hb_update)]

    
class DBN():
    """
    Implements a Deep Belief Network
    """
    
    def __init__(self, rbm_n_hidden=[100, 100], rbm_learning_rate = 5e-3,
                 rbm_batch_size = 100, rbm_n_epochs=500):
        
        pass
    
