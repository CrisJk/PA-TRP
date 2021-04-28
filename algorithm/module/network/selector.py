import tensorflow as tf
import config
import numpy as np
def __dropout__(x,keep_prob= 1.0):
    return tf.contrib.layers.dropout(x,keep_prob=keep_prob)

def __logit__(x,rel_tot):
    with tf.variable_scope('logit',reuse = tf.AUTO_REUSE):

        relation_matrix = tf.get_variable('relation_matrix',shape=[rel_tot,x.shape[1]],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias',shape=[rel_tot],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.matmul(x,tf.transpose(relation_matrix)) + bias

    return logit


def bag_average(x,scope,rel_tot,var_scope=None,dropout_before = False,keep_prob= 1.0):
    with tf.variable_scope("average",reuse=tf.AUTO_REUSE):
        if dropout_before:
            x = __dropout__(x,keep_prob)
        bag_repre = []
        for i in range(scope.shape[0]):
            bag_hidden_mat = x[scope[i][0]:scope[i][1]]
            bag_repre.append(tf.reduce_mean(bag_hidden_mat,0))

        bag_repre = tf.stack(bag_repre)
        if not dropout_before:
            bag_repre = __dropout__(bag_repre,keep_prob)

    return __logit__(bag_repre,rel_tot) , bag_repre


def _bag_attention_train_logit_(x,query,rel_tot):
    with tf.variable_scope('logit',reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix',[rel_tot,x.shape[1]],trainable=True,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        cur_relation = tf.nn.embedding_lookup(relation_matrix,query)
        att_logit = tf.reduce_sum(cur_relation*x,-1)
        return att_logit
def _bag_attention_test_logit_(x,rel_tot):
    with tf.variable_scope('logit',reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix',[rel_tot,x.shape[1]],trainable=False,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    attention_logit = tf.matmul(x,tf.transpose(relation_matrix))
    return attention_logit

def bag_attention(x,rel_tot,query,scope,train=True,keep_prob=1.0):
    print(x)
    with tf.variable_scope('attention',reuse=tf.AUTO_REUSE):
        if train:
            bag_attention_logit = _bag_attention_train_logit_(x,query,rel_tot)
            bag_repre = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                bag_attention_score = tf.nn.softmax(bag_attention_logit[scope[i][0]:scope[i][1]],-1)
                bag_attention_score = tf.expand_dims(bag_attention_score,0)
                iter_bag_repre = tf.squeeze(tf.matmul(bag_attention_score,bag_hidden_mat))
                bag_repre.append(iter_bag_repre)
            bag_repre = tf.stack(bag_repre)
            bag_repre = __dropout__(bag_repre,keep_prob)

            return __logit__(bag_repre,rel_tot)

        else:
            '''test'''
            bag_attention_logit = _bag_attention_test_logit_(x,rel_tot)
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                bag_attention_score = tf.nn.softmax(tf.transpose(bag_attention_logit[scope[i][0]:scope[i][1], :]), -1)


                iter_bag_repre = tf.matmul(bag_attention_score,bag_hidden_mat)
                iter_bag_logit =tf.diag_part(__logit__(iter_bag_repre,rel_tot))

                bag_logit.append(iter_bag_logit)

            bag_logit = tf.stack(bag_logit)

            return bag_logit


def sentence_attention(x,scope,rel_tot,keep_prob=1.0):
    with tf.variable_scope('attention',reuse=tf.AUTO_REUSE):

        d_a = 230
        d_r = 52
        bag_repre = []
        for i in range(scope.shape[0]):
            bag_mat = x[scope[i][0]:scope[i][1]] # (J * d_v) J is instances number in a bag

            att1 = tf.contrib.layers.fully_connected(bag_mat,d_a) #(J * d_a)
            A = tf.transpose(tf.contrib.layers.fully_connected(att1,d_r,activation_fn = None)) #(d_r * J)

            A_softmax = tf.nn.softmax(A,-1) #(d_r * J)

            A_softmax = tf.reduce_mean(A_softmax,0) # J

            M_L2 = tf.reduce_sum(tf.expand_dims(A_softmax,-1) * bag_mat , 0) #(d_v)

            bag_repre.append(M_L2)

        bag_repre = tf.stack(bag_repre)
        bag_repre = __dropout__(bag_repre, keep_prob)
        bag_logit = __logit__(bag_repre,rel_tot)


        return  bag_logit



def sentence_attention2(x,scope,rel_tot):

    hidden_size = config.model.gru_hidden_size

    with tf.variable_scope('sentence_attetion',reuse=tf.AUTO_REUSE):
        sen_a = tf.get_variable('attention_A', [hidden_size])
        sen_r = tf.get_variable('query_r', [hidden_size, 1])
        sen_d = tf.get_variable('bias_d', [rel_tot])
        relation_embedding = tf.get_variable('relation_embedding', [rel_tot, hidden_size])

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        sen_logit = []

        for i in range(scope.shape[0]):
            sen_repre.append(tf.tanh(x[scope[i][0]:scope[i][1]]))

            batch_size = scope[i][1] - scope[i][0]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [hidden_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [rel_tot]), sen_d))


            sen_logit.append(tf.nn.softmax(sen_out[i]))



    return sen_logit



def sentence_attention3(x,scope,rel_tot):
    with tf.variable_scope('sentence_attention',reuse=tf.AUTO_REUSE):
        hidden_size = config.model.gru_hidden_size*2
        sent_att_q = tf.get_variable('sen_att_q',[hidden_size,1],initializer=tf.contrib.layers.xavier_initializer())

        def getSentAtten(num):
            num_sents = num[1] - num[0]
            bag_sents = x[num[0]: num[1]]

            sent_atten_wts = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(bag_sents), sent_att_q), [num_sents]))

            bag_rep_ = tf.reshape(
                tf.matmul(
                    tf.reshape(sent_atten_wts, [1, num_sents]),
                    bag_sents),
                [hidden_size]
            )
            return bag_rep_

        bag_rep = tf.map_fn(getSentAtten, scope,dtype=tf.float32)

        with tf.variable_scope('FC1',reuse=tf.AUTO_REUSE) as scope:

            w_rel = tf.get_variable('w_rel', [hidden_size,rel_tot],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.001) )

            b_rel = tf.get_variable('b_rel', initializer=np.zeros([rel_tot]).astype(np.float32),regularizer=tf.contrib.layers.l2_regularizer(0.001))

            nn_out = tf.nn.xw_plus_b(bag_rep,w_rel,b_rel)


        return nn_out
