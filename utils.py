import numpy as np

def batch_generator(data, batch_size):
    """
    Generates batches of samples of size batch_size
    """
    data = np.array(data)
    n_batches = int(np.ceil(len(data) / float(batch_size)))
    
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        batch = data_shuffled[start:end]
        if len(batch) < batch_size:
            # Pad with zeros                                                               
            pad = np.zeros((batch_size - batch.shape[0], batch.shape[1]),
                           dtype=batch.dtype)
            batch = np.vstack((batch, pad))

        yield batch
