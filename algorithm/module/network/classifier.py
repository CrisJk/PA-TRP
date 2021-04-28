import tensorflow as tf
import numpy as np
def softmax_cross_entropy(x,label,rel_tot,weights_table = None,var_scope = None):
    with tf.variable_scope("loss",reuse = tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0

        else:
            weights = tf.nn.embedding_lookup(weights_table,label)

        label_onehot = tf.one_hot(indices=label,depth=rel_tot,dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot,logits=x,weights = weights)
        tf.summary.scalar("loss",loss)
        return loss


def binary_cross_entropy(x,label,weights_table = None):
    with tf.variable_scope("loss",reuse = tf.AUTO_REUSE):

        if weights_table is None:
            weights = 1.0

        else:
            weights = tf.nn.embedding_lookup(weights_table,label)

        one_label = tf.one_hot(indices=label,depth=2,dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_label,logits=x,weights=weights)

        return loss



def cross_entropy(x,label,rel_tot,weights_table = None,var_scope = None):

    with tf.variable_scope("loss",reuse= tf.AUTO_REUSE):
        EPS = 1e-9
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table,label)

        label_onehot = tf.one_hot(indices=label,depth=rel_tot,dtype=tf.float32)
        x = tf.clip_by_value(x,EPS,1-EPS)
        print(x)
        loss = tf.reduce_mean(-tf.reduce_sum( tf.expand_dims(weights,-1) * label_onehot * tf.math.log(x),-1))
        return loss
