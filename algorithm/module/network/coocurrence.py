import tensorflow as tf
import config
def __dropout__(x,keep_prob=1.0):
    return tf.contrib.layers.dropout(x,keep_prob=keep_prob)

def __logit__(e_embedding,rel_tot):
    with tf.variable_scope('deepwalk_logit',reuse=tf.AUTO_REUSE):
        h_1 = 100
        w = tf.get_variable('W',[e_embedding.shape[1],h_1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias',[h_1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        deepwalk_logit = tf.matmul(e_embedding,w) + bias

        deepwalk_logit = tf.nn.relu(deepwalk_logit)

        h_2 = 60
        logit_W2 = tf.get_variable('logit_W2', [deepwalk_logit.shape[1], h_2], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        logit_bias2 = tf.get_variable('logit_bias2', [h_2], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())

        deepwalk_logit = tf.matmul(deepwalk_logit,logit_W2) + logit_bias2
        deepwalk_logit = tf.nn.relu(deepwalk_logit)
        h_3 = rel_tot
        logit_W3 = tf.get_variable('logit_W3', [deepwalk_logit.shape[1], h_3], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        logit_bias3 = tf.get_variable('logit_bias3', [h_3], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
        deepwalk_logit = tf.matmul(deepwalk_logit,logit_W3) + logit_bias3

        return deepwalk_logit

def deepwalk(deepwalk_mat,entity1,entity2,scope,rel_tot,keep_prob=1.0):
    with tf.variable_scope("deepwalk_embedding",reuse=tf.AUTO_REUSE):
        deepwalk_embedding = tf.get_variable('deepwalk_mat',dtype=tf.float32,initializer=deepwalk_mat,trainable=False)
        bag_repre = []
        for i in range(scope.shape[0]):
            e1 = entity1[scope[i][0]]
            e2 = entity2[scope[i][0]]
            e1_embedding = tf.nn.embedding_lookup(deepwalk_embedding, e1) #(1,embedding_size)
            e2_embedding = tf.nn.embedding_lookup(deepwalk_embedding, e2) #(1,embedding_size)
            e_embedding = tf.add(-1.0*e1_embedding,e2_embedding)
            bag_repre.append(e_embedding)
        bag_repre = tf.stack(bag_repre)

        bag_repre = __dropout__(bag_repre,keep_prob)
        deepwalk_logit = __logit__(bag_repre,rel_tot)

        return bag_repre,deepwalk_logit

        #return bag_repre

def __entity_type_score__(entity_type_x,rel_tot):
    with tf.variable_scope('entity_type_logit',reuse=tf.AUTO_REUSE):
        h_1 = 250
        x = tf.contrib.layers.fully_connected(entity_type_x,h_1)
        h_2 = 100
        x = tf.contrib.layers.fully_connected(x,h_2)
        h_3 = rel_tot
        x = tf.contrib.layers.fully_connected(x,h_3,activation_fn=None)
        return x

def entity_type(entity_repre,entity1_type,entity2_type,entity1_type_mask,entity2_type_mask,entity1_type_length,entity2_type_length,scope,rel_tot,keep_prob=1.0):
    embedding_size = config.model.entity_type_embedding_size
    with tf.variable_scope('entity_type',reuse=tf.AUTO_REUSE):
        entity_type_embedding = tf.get_variable('entity_type_emebedding',[39,embedding_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        print("entity1_type",entity1_type)


        entity1_type_x = tf.multiply(tf.nn.embedding_lookup(entity_type_embedding, entity1_type),tf.expand_dims(1.0*entity1_type_mask,-1))

        entity1_type_x = tf.reduce_sum(entity1_type_x,-2)
        entity1_type_x = tf.multiply(entity1_type_x,tf.expand_dims(1.0/entity1_type_length,-1))



        entity2_type_x = tf.multiply(tf.nn.embedding_lookup(entity_type_embedding, entity2_type),
                                     tf.expand_dims(1.0 * entity2_type_mask, -1))
        entity2_type_x = tf.reduce_sum(entity2_type_x, -2)
        entity2_type_x = tf.multiply(entity2_type_x, tf.expand_dims(1.0/entity2_type_length, -1))

        entity_type_x = tf.concat((entity1_type_x,entity2_type_x),-1)

        entity_type_bag_x = []
        for j in range(scope.shape[0]):
            entity_type_bag_x.append(entity_type_x[scope[j][0]])

        entity_type_bag_x = tf.stack(entity_type_bag_x)
        entity_type_bag_x = __dropout__(entity_type_bag_x,keep_prob)
        entity_type_logit = __entity_type_score__(entity_type_bag_x,rel_tot)
        return entity_type_bag_x,entity_type_logit


def distance(deepwalk_mat,entity1,entity2,scope,rel_tot,relation_matrix0,relation_matrix1,relation_matrix2,keep_prob=1.0):
    with tf.variable_scope('relation_matrix',reuse=tf.AUTO_REUSE):
        rel_mat0 = tf.get_variable(name = 'rel_mat0',initializer=relation_matrix0,dtype=tf.float32,trainable=False)
        rel_mat1 = tf.get_variable(name= 'rel_mat1',initializer=relation_matrix1,dtype=tf.float32,trainable=False)
        rel_mat2 = tf.get_variable(name='rel_mat2',initializer=relation_matrix2,dtype = tf.float32,trainable=False)
        na_repre0 = tf.get_variable(name='na_repre0',shape = [1,rel_mat0.shape[1]],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
        # na_repre1 = tf.get_variable(name='na_repre1',shape = [1,rel_mat0.shape[1]],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
        # na_repre2 = tf.get_variable(name='na_repre2',shape = [1,rel_mat0.shape[1]],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)

        rel_mat0 = tf.concat([na_repre0,rel_mat0],0)
        rel_mat1 = tf.concat([na_repre0,rel_mat1],0)
        rel_mat2 = tf.concat([na_repre0,rel_mat2],0)


    with tf.variable_scope("distance",reuse=tf.AUTO_REUSE):
        deepwalk_embedding = tf.get_variable('deepwalk_mat',dtype=tf.float32,initializer=deepwalk_mat,trainable=False)
        cos_sim = []
        for i in range(scope.shape[0]):
            e1 = entity1[scope[i][0]]
            e2 = entity2[scope[i][0]]
            e1_embedding = tf.nn.embedding_lookup(deepwalk_embedding, e1) #(1,embedding_size)
            e2_embedding = tf.nn.embedding_lookup(deepwalk_embedding, e2) #(1,embedding_size)
            e_embedding = tf.nn.l2_normalize(e2_embedding-e1_embedding,0)

            rel_mat0 = tf.nn.l2_normalize(rel_mat0,-1)
            rel_mat1 = tf.nn.l2_normalize(rel_mat1,-1)
            rel_mat2 = tf.nn.l2_normalize(rel_mat2,-1)

            cos_sim0 = tf.reduce_sum(tf.multiply(e_embedding,rel_mat0),-1)
            cos_sim1 = tf.reduce_sum(tf.multiply(e_embedding, rel_mat1), -1)
            cos_sim2 = tf.reduce_sum(tf.multiply(e_embedding, rel_mat2), -1)
            iter_cos_sim = tf.concat([cos_sim0,cos_sim1,cos_sim2],-1)

            cos_sim.append(iter_cos_sim)

        cos_sim = tf.stack(cos_sim)
        cos_sim = __logit__(cos_sim, rel_tot)

        return cos_sim
