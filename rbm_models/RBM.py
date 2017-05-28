from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils import *


class RBM():
    """
    Implements a Restricted Boltzmann machine. Follows Sci-kit learn conventions,
    but fairly loosely.
    """
    
    def __init__(self, 
                 n_hidden=100, 
                 learning_rate = 5e-3,
                 batch_size = 100, 
                 n_epochs=500,
                 session = tf.Session(),
                 model_path="models/rbm.ckpt"):

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.sess = session
        self.model_path = model_path
        
        
    def __del__(self):
        self.sess.close()
        
        
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
        
        # Train the model
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #Run through all of the training data num_epochs times
        for epoch in tqdm(range(self.n_epochs)):
            for batch in batch_generator(X, self.batch_size):
                self.sess.run(self.update, feed_dict={self.x:batch})
                

    def sample(self, input):
        """
        Sample given visible units using Gibbs Sampling.
        """
        input = np.array(input) # convert to numpy in case
        sample = self.gibbs_sample(1).eval(session=self.sess,
                                           feed_dict={self.x: input})
                              
        return sample
    

    def compute_hidden(self, x):
        """
        Compute hidden units given visible units.
        """
        visible = tf.placeholder(tf.float32, [None, self.n_visible], name="visible")
        compute = sample(tf.sigmoid(tf.matmul(visible, self.W) + self.hb))

        h = self.sess.run(compute, feed_dict={visible:x})
        return h
    

    def compute_visible(self, h):
        """
        Compute visible units given hidden.
        """        
        hidden = tf.placeholder(tf.float32, [None, self.n_hidden], name="hidden")
        compute = sample(tf.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.vb))
        
        x = self.sess.run(compute, feed_dict={hidden:h})
        return x

    
    def gibbs_sample(self, k): # iterates for k steps
        """
        Run Gibbs Sampling for k iterations 
        """
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

    
    def gibbs_sample_converge(self):
        """
        Run Gibbs Sampling until convergence
        """
        def step(x, stop_condition):
            x_prev = x
            x = sample(x)
            #Propagate the visible values to sample the hidden values
            h_k = sample(tf.sigmoid(tf.matmul(x, self.W) + self.hb))
            #Propagate the hidden values to sample the visible values
            x = tf.sigmoid(tf.matmul(h_k, tf.transpose(self.W)) + self.vb)
            
            # Convergence of probability vectors
            stop_condition = (tf.reduce_mean(tf.square(x - x_prev)) > 0.2)
            return x, stop_condition

        [x_sample, _] = tf.while_loop(lambda x, stop_condition: stop_condition,
                                      step, [self.x, tf.constant(True)], 
                                      parallel_iterations=1,
                                      back_prop = False)

        x_sample = sample(x_sample)
        # TF tutorials said we need this to stop RBM values from backpropogating
        x_sample = tf.stop_gradient(x_sample) 
        return x_sample
    
    
    def get_free_energy_cost(self):
        """
        Get free energy cost.
        """
        self.x  = tf.placeholder(tf.float32, [None, self.n_visible], name="x") 

        x_sample = self.gibbs_sample(1)
        
        def F(xx):
            # Computes free energy of input
            return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(xx, self.W) + self.hb)), 1) 
        - tf.matmul(xx, tf.transpose(self.vb))
        
        # Calculates different in free energy
        cost =  tf.reduce_mean(tf.sub(F(self.x), F(x_sample)))
        return cost
    
    
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

        #When we do sess.run(update), TensorFlow will run all 3 update steps
        self.update = [self.W.assign_add(W_update), self.vb.assign_add(vb_update), 
                       self.hb.assign_add(hb_update)]

        
    def _train_model_down(self):
        """
        Run contrastive divergence to get weight updates in a top-down
        pass for single epoch.
        """
        self.h  = tf.placeholder(tf.float32, [None, self.n_hidden], name="h") 
        
        # Run gibbs sampling for one step and save samples for x and h
        x = sample(tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.vb))
        
        h_sample = sample(tf.sigmoid(tf.matmul(x, self.W) + self.hb))
        x_sample = sample(tf.sigmoid(tf.matmul(h_sample, tf.transpose(self.W)) + self.vb))
        
        #Update the values of W, hb, and vb
        size_h = tf.cast(tf.shape(self.h)[0], tf.float32)

        W_update  = tf.mul(self.learning_rate/size_h, 
                           tf.sub(tf.matmul(tf.transpose(x), self.h), \
                                      tf.matmul(tf.transpose(x_sample), h_sample)))
        vb_update = tf.mul(self.learning_rate/size_h, 
                           tf.reduce_sum(tf.sub(x, x_sample), 0, True))
        hb_update = tf.mul(self.learning_rate/size_h, 
                           tf.reduce_sum(tf.sub(self.h, h_sample), 0, True))

        #When we do sess.run(update), TensorFlow will run all 3 update steps
        self.update_down = [self.W.assign_add(W_update), self.vb.assign_add(vb_update), 
                            self.hb.assign_add(hb_update)]

    
    def transform(self, X):
        """
        Tune parameters of fitted model on new data.
        """
        # Get training process
        self._train_model()

        for batch in batch_generator(X, self.batch_size):
            self.sess.run(self.update, feed_dict={self.x:batch})
        
        
    def transform_down(self, hidden):
        """
        Tune parameters of fitted model on new data in downward 
        direction..
        """
        self._train_model_down()
                
        for batch in batch_generator(hidden, self.batch_size):
            self.sess.run(self.update_down, feed_dict={self.h:hidden})


