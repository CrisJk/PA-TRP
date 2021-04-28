import config
import tensorflow as tf
import module
import numpy as np
class model:

    def __init__(self,data_loader,batch_size,max_length,encoder,selector,entity_inf,dataset):
        self.encoder = encoder
        self.selector = selector
        self.word = tf.placeholder(dtype=tf.int32,shape=[None,max_length], name = 'word')
        self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None,max_length], name = 'pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None,max_length], name = 'pos2')
        self.label = tf.placeholder(dtype=tf.int32,shape=[batch_size], name = 'label')
        self.label2 = tf.placeholder(dtype=tf.int32,shape=[batch_size] ,name = 'label2')
        self.label3 = tf.placeholder(dtype=tf.int32,shape=[batch_size],name = 'label3')

        self.ins_label = tf.placeholder(dtype=tf.int32,shape=[None],name = 'ins_label')
        self.ins_label2 = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label2')
        self.ins_label3 = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label3')

        self.length = tf.placeholder(dtype=tf.int32,shape=[None],name='length')
        self.scope = tf.placeholder(dtype=tf.int32,shape=[batch_size,2],name='scope')
        self.data_loader = data_loader

        self.rel_tot = data_loader.rel_tot
        self.rel_tot2 = data_loader.rel_tot2
        self.rel_tot3 = data_loader.rel_tot3

        self.rel_rel2 = data_loader.rel_rel2
        self.rel_rel3 = data_loader.rel_rel3

        self.word_vec_mat = data_loader.word_vec_mat

        self.mask = tf.placeholder(dtype=tf.int32,shape=[None,max_length],name='mask')

        self.bc_embedding = tf.placeholder(dtype=tf.float32,shape=[None,max_length,768],name='bert_embedding')
        if config.model.use_coocurrence == True:
            self.coocurrence_embedding = data_loader.coocurrence_embedding
        else:
            self.coocurrence_embedding = None

        if config.model.use_deepwalk == True:
            self.deepwalk_mat = data_loader.deepwalk_mat
        else:
            self.deepwalk_mat = None
        self.deepwalk_mat = data_loader.deepwalk_mat

        self.entity1 = tf.placeholder(dtype = tf.int32,shape=[None],name='entity1')
        self.entity2 = tf.placeholder(dtype=tf.int32,shape=[None],name='entity2')

        self.entity1_type = tf.placeholder(dtype=tf.int32, shape=[None,None],name='entity1_type')
        self.entity2_type = tf.placeholder(dtype=tf.int32, shape=[None,None],name='entity2_type')

        self.entity1_type_mask = tf.placeholder(dtype=tf.float32,shape=[None,None],name='entity1_type_mask')
        self.entity2_type_mask = tf.placeholder(dtype=tf.float32,shape=[None,None],name='entity2_type_mask')

        self.entity1_type_length = tf.placeholder(dtype=tf.float32,shape=[None],name='entity1_type_length')
        self.entity2_type_length = tf.placeholder(dtype=tf.float32,shape=[None],name='entity2_type_length')

        import numpy
        relation_matrix0 = np.load(config.dir.dataset_dir[dataset]['rel_embedding'])
        relation_matrix1 = np.load(config.dir.dataset_dir[dataset]['rel_embedding2'])
        relation_matrix2 = np.load(config.dir.dataset_dir[dataset]['rel_embedding3'])

        # Embedding layer
        x = module.network.embedding.word_position_embedding(self.word,self.word_vec_mat,self.pos1,self.pos2)

        # Encoder
        if encoder == 'pcnn':
            x_train = module.network.encoder.pcnn(x,self.mask,keep_prob = 0.5)
            x_test = module.network.encoder.pcnn(x,self.mask,keep_prob = 1.0)
        elif encoder == 'bigru_att':
            x_train,penalisation,self.A_softmax = module.network.encoder.bigru_att(x,self.mask,self.length,0.5)
            x_test,penalisation,self.A_softmax = module.network.encoder.bigru_att(x,self.mask,self.length,1.0)
        elif encoder == 'cnn':
            x_train = module.network.encoder.cnn(x, keep_prob=0.5)
            x_test = module.network.encoder.cnn(x, keep_prob=1.0)
        elif encoder == 'rnn':
            x_train = module.network.encoder.rnn(x, self.length, keep_prob=0.5)
            x_test = module.network.encoder.rnn(x, self.length, keep_prob=1.0)
        else:
            raise NameError
        #Selector
        #print(self.selector)
        if self.selector == 'ave':
            self._train_logit , train_repre = module.network.selector.bag_average(x_train,self.scope,self.rel_tot,keep_prob=0.5)
            self._test_logit,test_repre = module.network.selector.bag_average(x_test,self.scope,self.rel_tot)

        if self.selector == 'att':
            self._train_logit = module.network.selector.bag_attention(x_train,self.rel_tot,self.ins_label,self.scope,train=True,keep_prob=0.5)
            self._test_logit = module.network.selector.bag_attention(x_test,self.rel_tot,self.ins_label,self.scope,train=False,keep_prob=1.0)

        if entity_inf == 'tmr':

            with tf.variable_scope('weight', reuse=tf.AUTO_REUSE):
                self.alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=0.33, trainable=True)
                self.beta = tf.get_variable('beta', dtype=tf.float32, initializer=0.33, trainable=True)
                self.gamma = tf.get_variable('gamma', dtype=tf.float32, initializer=0.33, trainable=True)

            self.train_entity_repre, self.train_entity_logit = module.network.coocurrence.deepwalk(self.deepwalk_mat,
                                                                                                   self.entity1,
                                                                                                   self.entity2,
                                                                                                   self.scope,
                                                                                                   self.rel_tot,
                                                                                                   keep_prob=0.5)
            self.test_entity_repre, self.test_entity_logit = module.network.coocurrence.deepwalk(self.deepwalk_mat,
                                                                                                 self.entity1,
                                                                                                 self.entity2,
                                                                                                 self.scope,
                                                                                                 self.rel_tot,
                                                                                                 keep_prob=1.0)


            self.train_entity_type_repre, self.train_entity_type_logit = module.network.coocurrence.entity_type(
                None, self.entity1_type, self.entity2_type, self.entity1_type_mask,
                self.entity2_type_mask, self.entity1_type_length, self.entity2_type_length, self.scope, self.rel_tot,
                keep_prob=0.5)
            self.test_entity_type_repre, self.test_entity_type_logit = module.network.coocurrence.entity_type(
                None, self.entity1_type, self.entity2_type, self.entity1_type_mask,
                self.entity2_type_mask, self.entity1_type_length, self.entity2_type_length, self.scope, self.rel_tot,
                keep_prob=1.0)

            self._train_logit = self.alpha * self._train_logit + self.beta * self.train_entity_logit + self.gamma*self.train_entity_type_logit
            self._test_logit = self.alpha * self._test_logit  + self.beta * self.test_entity_logit + self.gamma*self.test_entity_type_logit

            h_1 = 310
            h_2 = 510
            h_3 = 310
            h_4 = self.rel_tot
            with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self._train_logit.shape[-1], h_1], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias1 = tf.get_variable('bias1', [h_1], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w2 = tf.get_variable('w2', [h_1, h_2], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias2 = tf.get_variable('bias2', [h_2], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w3 = tf.get_variable('w3', [h_2, h_3], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias3 = tf.get_variable('bias3', [h_3], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w4 = tf.get_variable('w4', [h_3, h_4], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias4 = tf.get_variable('bias4', [h_4], dtype=tf.float32,

                                        initializer=tf.contrib.layers.xavier_initializer())

            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w1) + bias1)
            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w2) + bias2)
            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w3) + bias3)
            self._train_logit = tf.matmul(self._train_logit, w4) + bias4

            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w1) + bias1)
            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w2) + bias2)
            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w3) + bias3)
            self._test_logit = tf.matmul(self._test_logit, w4) + bias4

        if entity_inf == 'trp':
            with tf.variable_scope('weight', reuse=tf.AUTO_REUSE):
                self.alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=0.5, trainable=False)
                self.beta = tf.get_variable('beta', dtype=tf.float32, initializer=0.5, trainable=False)

            self.train_sim_logit = module.network.coocurrence.distance(self.deepwalk_mat,self.entity1,self.entity2,
                                                                       self.scope,self.rel_tot,relation_matrix0,
                                                                       relation_matrix1,relation_matrix2,keep_prob=0.5)
            self.test_sim_logit = module.network.coocurrence.distance(self.deepwalk_mat,self.entity1,self.entity2,
                                                                       self.scope,self.rel_tot,relation_matrix0,
                                                                       relation_matrix1,relation_matrix2,keep_prob=1.0)
            self.train_entity_type_repre, self.train_entity_type_logit = module.network.coocurrence.entity_type(None, self.entity1_type,
                                                                                                                self.entity2_type, self.entity1_type_mask,
                                                                                                                self.entity2_type_mask, self.entity1_type_length,
                                                                                                                self.entity2_type_length, self.scope, self.rel_tot,
                                                                                                                keep_prob=0.5)
            self.test_entity_type_repre, self.test_entity_type_logit = module.network.coocurrence.entity_type(None, self.entity1_type, self.entity2_type,
                                                                                                              self.entity1_type_mask,self.entity2_type_mask,
                                                                                                              self.entity1_type_length, self.entity2_type_length,
                                                                                                              self.scope, self.rel_tot,keep_prob=1.0)

            self._train_logit = tf.concat([self._train_logit,self.train_sim_logit,self.train_entity_type_logit],-1)
            self._test_logit = tf.concat([self._test_logit,self.test_sim_logit,self.test_entity_type_logit],-1)

            h_1 = 120
            h_2 = 300
            h_3 = 120
            h_4 = self.rel_tot
            with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self._train_logit.shape[-1], h_1], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias1 = tf.get_variable('bias1', [h_1], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w2 = tf.get_variable('w2', [h_1, h_2], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias2 = tf.get_variable('bias2', [h_2], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w3 = tf.get_variable('w3', [h_2, h_3], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias3 = tf.get_variable('bias3', [h_3], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
                w4 = tf.get_variable('w4', [h_1, h_4], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias4 = tf.get_variable('bias4', [h_4], dtype=tf.float32,

                                        initializer=tf.contrib.layers.xavier_initializer())

            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w1) + bias1)
            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w2) + bias2)
            self._train_logit = tf.nn.tanh(tf.matmul(self._train_logit, w3) + bias3)
            self._train_logit = tf.matmul(self._train_logit, w4) + bias4

            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w1) + bias1)
            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w2) + bias2)
            self._test_logit = tf.nn.tanh(tf.matmul(self._test_logit, w3) + bias3)
            self._test_logit = tf.matmul(self._test_logit, w4) + bias4

        #classifier
        self._loss = module.network.classifier.softmax_cross_entropy(self._train_logit,self.label,self.rel_tot,weights_table=self.get_weights())

        self._test_loss = module.network.classifier.softmax_cross_entropy(self._test_logit, self.label, self.rel_tot)
        self._test_logit = tf.nn.softmax(self._test_logit,-1)

    def loss(self):
        return self._loss
    def interval(self):
        return tf.constant(1),tf.constant(2)
    def test_loss(self):
        return self._test_loss
    def train_logit(self):
        return self._train_logit
    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.data_loader.data_rel)):
                _weights_table[self.data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table
