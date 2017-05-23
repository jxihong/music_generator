import tensorflow as tf
import numpy as np

def sample(probs):
    """
    Takes in a vector of probabilities, and returns a random vector of 0s and 1s 
    sampled from the input vector
    """
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


def gibbs_sample(x, W, bv, bh, k):
    """
    Runs Gibbs Sampling for k steps from the probability distribution of the RBM defined by W, bh, bv
    """
    def step(i, k, x_k):
        """
        Run a single step
        """
        #Propagate the visible values to get the hidden values
        h_k = sample(tf.sigmoid(tf.matmul(x_k, W) + bh))
        #Propagate the hidden values to get the visible values
        x_k = sample(tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + bv))
        return i + 1, k, x_k
    
    i = tf.constant(0) #counter
    [_, _, x_sample] = tf.while_loop(lambda i, k, *args: i < k, step,
                                     [i, tf.constant(k), x], 
                                     parallel_iterations=1, 
                                     back_prop=False)
    
    # TF tutorials said we need this to stop RBM values from backpropogating
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample


def gibbs_sample_converge(x, W, bv, bh):
    """
    Run Gibbs Sampling until convergence on RBM.
    """
    def step(x, stop_condition):
        x_prev = x
        #Propagate the visible values to sample the hidden values
        h_k = sample(tf.sigmoid(tf.matmul(x, W) + bh))
        #Propagate the hidden values to sample the visible values
        x = tf.sigmoid(tf.matmul(h_k, tf.transpose(W)) + bv)
            
        # Convergence of probability vectors
        stop_condition = (tf.reduce_mean(tf.square(x - x_prev)) > 0.2)
        return x, stop_condition

    [x_sample, _] = tf.while_loop(lambda x, stop_condition: stop_condition,
                                  step, [x, tf.constant(True)], 
                                  parallel_iterations=1,
                                  back_prop = False)
    
    x_sample = sample(x_sample)
    # TF tutorials said we need this to stop RBM values from backpropogating
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample



def cd_update(x, W, bv, bh, learning_rate=5e-3):
    """
    Get updates from contrastive divergence for a single
    epoch of training.
    """
    # Run gibbs sampling for one step and save samples for x and h
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
        
    x_sample = sample(tf.sigmoid(tf.matmul(h, tf.transpose(W)) + bv))
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    #Update the values of W, bh, and bv
    size_x = tf.cast(tf.shape(x)[0], tf.float32)

    W_update  = tf.mul(learning_rate/size_x, 
                       tf.sub(tf.matmul(tf.transpose(x), h), \
                                  tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_update = tf.mul(learning_rate/size_x, 
                       tf.reduce_sum(tf.sub(x, x_sample), 0, True))
    bh_update = tf.mul(learning_rate/size_x, 
                       tf.reduce_sum(tf.sub(h, h_sample), 0, True))
    
    #When we do sess.run(update), TensorFlow will run all 3 update steps
    update = [W.assign_add(W_update), bv.assign_add(bv_update), 
              bh.assign_add(bh_update)]
    return update


def get_free_energy_cost(x, W, bv, bh, k):   
    """
    Get free energy cost of a sample.
    """
    #First, draw a sample from the RBM
    x_sample = gibbs_sample(x, W, bv, bh, k)

    def free_energy(v):
        #The function computes the free energy of a visible vector. 
        return (-tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(v, W) + bh)), 1) 
                 - tf.matmul(v, tf.transpose(bv)))

    #The cost is based on the difference in free energy between x and xsample
    cost = tf.reduce_mean(tf.sub(free_energy(x), free_energy(x_sample)))
    return cost
