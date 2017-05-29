# Rohan CHoudhury
# this is the seq2seq model to run on Doshis GPU

from keras.models import Sequential
import time
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Activation
from keras.models import Model
from utils import *
import glob
import numpy as np
from midi_parser import *

from rnn import *

NUM_NOTES = 156
HIDDEN_SIZE = 500 # arbitrary
LAYERS = 2
N_EPOCHS = 200

# First, import the data. This will take a while.
print("Importing Data...")
start = time.time()
bass, melodies = read_state_matrices()
X_dat = melodies
y_dat = bass
end = time.time()
print("Importing took " + str(end-start) + " seconds.")


# Prepare the data.
X = []
y = []
window = 5
for j in range(len(X_dat)):
    for i in range(1, len(X_dat[j]), window):
        if (i + window >= len(X_dat[j])):
            break
        X.append(X_dat[j][i: i + window])
        y.append(y_dat[j][i + window])

#model = build_model(window, n_visible)

X = np.array(X)
y = np.array(y)

print X.shape
print y.shape

# Build and compile the model. 
model = build_model(window, NUM_NOTES)

print(model.summary())

# Save the model as a json, and serialize the fitting. 
# Train the model and save it at checkpoints, returning the final thing
# at the end.  
model_json = model.to_json()
with open("parameter_checkpoints/rnn.json", "w") as json_file:
    json_file.write(model_json)

for i in range(4):
    model.fit(X, y, batch_size=100, nb_epoch=50)
    
    print("Saved checkpoint after {} epochs.".format(i * 50))
    # serialize weights to HDF5
    model.save_weights("parameter_checkpoints/seq2seq_bass_epoch_{}.h5".format((i + 1) * 50))
        
# serialize weights to HDF5
model.save_weights("parameter_checkpoints/seq2seq_bass_final.h5")

