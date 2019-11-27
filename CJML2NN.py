# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:18:18 2019

@author: yimeng
"""

# 1. import modules
from keras.legacy import interfaces
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Input, Flatten, Activation, Dot, Multiply, dot
from tensorflow.keras.constraints import Constraint, MaxNorm
from tensorflow.keras import Model
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# 2. define the model
def mf(n_person=100, n_item=3000, para_dim = 5):
    """
    Input: dimensions of person-item matrix
    Output: neural network with multiple inputs and embedding layers
    """
    p = Input(shape=[1], name='person')
    p_e = Embedding(n_person , para_dim, embeddings_initializer='RandomNormal', name='person_embedding')(p)
    p_e = MaxNorm(max_value=5*np.sqrt(para_dim))(p_e) # maxnorm

    i = Input(shape=[1], name='item')
    i_e = Embedding(n_item, para_dim, embeddings_initializer='RandomNormal', name='item_embedding')(i)
    i_e = MaxNorm(max_value=5*np.sqrt(para_dim))(i_e) # maxnorm
    
    d = Input(shape=[1], name='residual')
    d_e = Embedding(n_item, 1, embeddings_initializer='RandomNormal', name = 'res_embed')(i)
    d_e = MaxNorm(max_value=5*np.sqrt(para_dim))(d_e) # maxnorm

    output = Dot(axes=-1, name='dotProduct')([p_e, i_e]) + d_e
#     print(output.shape)
    output = Flatten(name='output')(output)
    main_output = Activation('sigmoid')(output)
#     print(main_output.shape)
    model = Model([p, i, d], main_output)
    return model

# 3. process the data
def DataProcessing(person_item_matrix):
    """
    Input: Person-Item Matrix
    Output: 
    Indecies for each trainging responce (p_train, i_train)
    Indecies for each validation responce (p_val, i_val)
    Responce (y_train, y_val)
    """
    row = list(range(person_item_matrix.shape[0]))  # index of row
    column = list(range(person_item_matrix.shape[1]))  # index of column
    all = [row, column]
    combine_index = list(itertools.product(*all))  # combination
    idx = ['input {}'.format(i) for i in range(1, len(combine_index)+1)]
    df = pd.DataFrame(combine_index, index = idx, columns = ['row', 'column'])  # all permutations for row-column index
    row_input = df["row"].tolist()
    col_input = df["column"].tolist()
    
    # data structure
    person_input = [i for i in row_input]
    item_input = [i for i in col_input]
    output = person_item_matrix.tolist()
    flt_output_train  = list(itertools.chain(*output))
    output = [i for i in flt_output_train]
    
    # split trainingset and testset with portion 8:2
    p_train, p_val, i_train, i_val, y_train, y_val = train_test_split(person_input, item_input, output, test_size=0.2, random_state=42)
    return p_train, p_val, i_train, i_val, y_train, y_val

# 4. preprocess real data: missing data / low voting rate
from numpy import loadtxt
org_data = loadtxt('sen108kh.csv', delimiter=',')  # original person-item matrix with 1,6,9
rows = (org_data == 9).sum(1)  # aggragate the number of not voting rach person
data_clean = org_data[np.where(rows/org_data.shape[1] < 0.05)]  # remove the person record with more than 5% not voting 
subs = np.zeros(shape = data_clean.shape)
myseed = 0
for i in range(data_clean.shape[0]):
    for j in range(data_clean.shape[1]):
        np.random.seed(myseed)
        subs[i, j] = np.random.randint(0, 2)  # random int from {0,1}
        myseed += 1

data_clean[data_clean == 9] = subs[data_clean == 9]  # replace 9 in original data with randomly generated 0 and 1 (missing data)
data_clean[data_clean == 6] = 0  # replace 6 in original data with 0 (oppose)

p_train, p_val, i_train, i_val, y_train, y_val = DataProcessing(data_clean) # processed to be training and validation data

# 5. SGD model
from numpy.random import seed
seed(20191020)
import tensorflow as tf
tf.random.set_seed(20191020)

from tensorflow.keras.optimizers import SGD
model = mf(*data_clean.shape, para_dim = 2)
sgd = SGD(lr = 0.05, decay=0.0, momentum=0.99, nesterov=False)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)
hist_sgd = model.fit([p_train, i_train, i_train], y_train, 
          validation_data=([p_val, i_val, i_val], y_val),
         batch_size = 56, epochs=50)

# 6. Alternating Minimization
class AlterHacker(keras.callbacks.Callback):
    def __init__(self, model, *args, **kwargs):
        super(AlterHacker, self).__init__(*args, **kwargs)
        self.model = model
        self.iteration = 0
        self._cache_weights = None
        self.param_update_order = ([0], [1,2])
        
    def on_train_batch_begin(self, batch, logs={}):
        self._cache_weights = self.model.get_weights()

    def on_train_batch_end(self, batch, logs={}):
        """
        Set part of the weights to values in last interation after training for the new interation for each batch
        """
        cur_params = self.param_update_order[self.iteration % len(self.param_update_order)]
        now_weights = self.model.get_weights()
        for i, w in enumerate(self.model.weights):
            if i not in cur_params:
                now_weights[i] = self._cache_weights[i]
        self.model.set_weights(now_weights)
        self.iteration += 1
    def on_test_batch_begin(self, batch, logs={}):
        return
    def on_test_batch_end(self, batch, logs={}):
        return
    def on_test_begin(self, logs={}):
        return
    def on_test_end(self, logs={}):
        return
    
# 7. Alternating model
seed(20191020)
tf.random.set_seed(20191020)
import keras
import tensorflow.keras

model = mf(*data_clean.shape, para_dim = 2)
alter_log = AlterHacker(model)

sgd = SGD(lr = 0.05, decay=0.0, momentum=0.99, nesterov=False)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)
hist_alter = model.fit([p_train, i_train, i_train], y_train, 
          validation_data=([p_val, i_val, i_val], y_val),
         batch_size = 56, epochs=50,
         callbacks=[alter_log])

# 8. SGD VS Alternating on real data
import matplotlib.pyplot as plt

plt.plot(hist_sgd.history['val_accuracy'])
plt.plot(hist_alter.history['val_accuracy'])
plt.axhline(0.9178295, ls = '--', color = 'r')  # compare with CJML result
plt.legend(['SGD','Alternating','CJML'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.show();
