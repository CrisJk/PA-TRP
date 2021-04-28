import sys
sys.path.append("..")
import config
import tensorflow as tf
import numpy as np

def __cnn_cell__(x,hidden_size,kernel_size,stride_size):
    x = tf.layers.conv1d(inputs=x,
                         filters = hidden_size,
                         kernel_size=kernel_size,
                         strides = stride_size,
                         padding = 'same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

def __dropout__(x,keep_prob= 1.0):
    return tf.contrib.layers.dropout(x,keep_prob=keep_prob)

def __piecewise_pooling(x,mask):
    mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype = np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding,mask)
    hidden_size = x.shape[-1]

    x = tf.reduce_max(tf.expand_dims(mask*1000,2) + tf.expand_dims(x,3) , axis=1) - 1000

    return tf.reshape(x,[-1,hidden_size*3])

def pcnn(x,mask,keep_prob,hidden_size= config.model.pcnn_hidden_size,kernel_size = config.model.pcnn_kernel_size,stride_size = config.model.pcnn_stride_size,
         activiation=config.model.pcnn_activiation,var_scope=None):
    with tf.variable_scope("pcnn", reuse = tf.AUTO_REUSE):
        x = __cnn_cell__(x,hidden_size,kernel_size,stride_size)
        x = __piecewise_pooling(x,mask)
        x = activiation(x)
        x = __dropout__(x, keep_prob=keep_prob)
        return x

def bgwa(x,mask,length,keep_prob,activiation=config.model.bgwa_activiation_activiation):
    hidden_size = config.model.gru_hidden_size
    with tf.variable_scope('bigru', reuse=tf.AUTO_REUSE):
        fw_gru = tf.nn.rnn_cell.GRUCell(hidden_size, bias_initializer=tf.contrib.layers.xavier_initializer())
        bw_gru = tf.nn.rnn_cell.GRUCell(hidden_size, bias_initializer=tf.contrib.layers.xavier_initializer())
        gru_output, gru_state = tf.nn.bidirectional_dynamic_rnn(fw_gru, bw_gru, x, sequence_length=length,
                                                                dtype=tf.float32)

        gru = tf.concat(gru_output, 2)  # (N*seq_len*(2*h_size) )

    with tf.variable_scope("wordlevel_A", reuse=tf.AUTO_REUSE):
        attention_matrix = tf.contrib.layers.fully_connected(gru,2 * hidden_size, biases_initializer=None,activation_fn=tf.nn.tanh)
        print(attention_matrix)
        attention_vector = tf.squeeze(tf.contrib.layers.fully_connected(attention_matrix, 1,biases_initializer=None, activation_fn=None),
                                      -1)

        paddings = tf.ones_like(mask) * (-2 ** 30 + 1)
        paddings = tf.cast(paddings, tf.float32)

        attention_value = tf.where(mask>0,attention_vector,paddings)
        attention_values_softmax = tf.expand_dims(tf.nn.softmax(attention_value, -1), -1)
        attention_embedding = gru * attention_values_softmax
        print(attention_embedding)
        x = __piecewise_pooling(attention_embedding, mask)
        x = __dropout__(x, keep_prob=keep_prob)

    return x



def eta(x,mask,length,e1_embedding,e2_embedding,keep_prob):
    hidden_size = config.model.gru_hidden_size
    with tf.variable_scope('bigru',reuse=tf.AUTO_REUSE):
        fw_gru = tf.nn.rnn_cell.GRUCell(hidden_size,bias_initializer=tf.contrib.layers.xavier_initializer())
        bw_gru = tf.nn.rnn_cell.GRUCell(hidden_size,bias_initializer=tf.contrib.layers.xavier_initializer())

        gru_output,gru_state = tf.nn.bidirectional_dynamic_rnn(fw_gru,bw_gru,x,sequence_length=length,dtype=tf.float32)

        gru = tf.concat(gru_output,2)

        gru_tanh = tf.nn.tanh(gru)


    with tf.variable_scope('entity_attention',reuse=tf.AUTO_REUSE):
        e1_embedding = tf.contrib.layers.fully_connected(e1_embedding,2*hidden_size,activation_fn=tf.nn.tanh)
        e2_embedding = tf.contrib.layers.fully_connected(e2_embedding,2*hidden_size,activation_fn=tf.nn.tanh)

        attention_score = (tf.nn.softmax(tf.matmul(gru_tanh,tf.expand_dims(e1_embedding,-1)),-2)+\
                          tf.nn.softmax(tf.matmul(gru_tanh,tf.expand_dims(e2_embedding,-1)),-2))/2.0

        x = gru*attention_score
        #x = __piecewise_pooling(x,mask)
        #x = tf.reduce_sum(x,-2)
        x = tf.reduce_sum(x,-1)
        x = tf.nn.tanh(x)
        x = __dropout__(x,keep_prob=keep_prob)

    return x


def bigru_att(x,mask,length,keep_prob):
    '''
    Args:
        x: sentence embedding, a tensor with shape of (N,seq_len,embedding_size)
        length: sequence length, a 1D tensor with shape of (N)

    Return:
        sentence encoding, a tensor with shape of (N,encoding_size)
    '''

    #bigru
    hidden_size = config.model.gru_hidden_size

    with tf.variable_scope('bigru',reuse=tf.AUTO_REUSE):
        fw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        gru_output,gru_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,x,sequence_length=length,dtype=tf.float32)
        gru = tf.concat(gru_output,2) #(N, seq_len, 2*h_size)
        gru_tanh = tf.nn.tanh(gru) #(N,seq_len, 2*h_size)


    d_a = 120
    d_r = 100
    d_v = 230
    with tf.variable_scope('word_attention',reuse=tf.AUTO_REUSE):

        w_o = tf.get_variable('w_o',shape=[d_r*2*hidden_size,d_v],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32) #(d_r*2*h_size , d_v)

        att1 = tf.contrib.layers.fully_connected(gru,d_a,biases_initializer=None,activation_fn=tf.nn.tanh) #(N,seq_len,d_a)

        A = tf.transpose(tf.contrib.layers.fully_connected(att1,d_r,biases_initializer=None),[0,2,1]) #(N,d_r,seq_len)

        mask = tf.tile(tf.expand_dims(mask,1),[1,d_r,1]) #(N,d_r,seq_len)
        paddings = tf.ones_like(mask) * (-2 ** 30 + 1)
        paddings = tf.cast(paddings, tf.float32)
        A = tf.where(mask > 0, A, paddings) #(N,d_r,seq_len)'


        A_softmax = tf.nn.softmax(A,-1) #(N,d_r,seq_len)

        M_L1 = tf.matmul(A_softmax,gru) #(N,d_r,2*h_size)

        output = tf.nn.relu(tf.matmul(tf.reshape(M_L1,[-1,d_r*2*hidden_size]) , w_o) ) #(N*d_v)
        output = __dropout__(output,keep_prob)

    penalisation = tf.norm(tf.matmul(A_softmax,tf.transpose(A_softmax,[0,2,1])) - tf.eye(d_r),2)
    return output,penalisation,A_softmax

def __pooling__(x):
    return tf.reduce_max(x, axis=-2)

def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __pooling__(x)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x



def __rnn_cell__(hidden_size, cell_name='lstm'):
    if isinstance(cell_name, list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return __rnn_cell__(hidden_size, cell_name[0])
        cells = [__rnn_cell__(hidden_size, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name.lower() == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_size)
    raise NotImplementedError


def rnn(x, length, hidden_size=230, cell_name='gru', var_scope=None, keep_prob=1.0):
    hidden_size = config.model.gru_hidden_size
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        cell = __rnn_cell__(hidden_size, cell_name)

        _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32)
        if isinstance(states, tuple):
            states = states[0]
        return states






