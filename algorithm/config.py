# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
# 以instance作为最小单位
MODE_INSTANCE = 0
# 以bag为最小单位，每个bag中实体对相同，一般用于test(因为不知道relation)
MODE_ENTPAIR_BAG = 1
# 以bag为最小单位，每个bag中实体对和关系相同，一般用于train
MODE_RELFACT_BAG = 2

root_path = os.getcwd()
print(root_path)
class dir:
    dataset_dir = {
        "nyt": {
            "root":os.path.join(root_path,"data/nyt/"),
            "train":os.path.join(root_path,"data/nyt/train.json"),
            "test":os.path.join(root_path,"data/nyt/test.json"),
            "rel2id":os.path.join(root_path,"data/nyt/rel2id.json"),
            "rel2id2": os.path.join(root_path, "data/nyt/rel2id2.json"),
            "rel2id3": os.path.join(root_path, "data/nyt/rel2id3.json"),
            "rel_rel2": os.path.join(root_path, "data/nyt/rel_rel2.json"),
            "rel_rel3": os.path.join(root_path, "data/nyt/rel_rel3.json"),
            "word2vec":os.path.join(root_path,"data/nyt/word_vec.json"),
            "entity2id":os.path.join(root_path,"data/nyt/entity2id.json"),
            "rel_embedding":os.path.join(root_path,"data/nyt/relation_matrix.npy"),
            "rel_embedding2": os.path.join(root_path, "data/nyt/relation_matrix2.npy"),
            "rel_embedding3": os.path.join(root_path, "data/nyt/relation_matrix3.npy"),
            "deepwalk":os.path.join(root_path,"data/nyt/normalized_all.txt"),
            "entitytype":os.path.join(root_path,"data/nyt/coarse_type_info.json"),
            "mid2id":os.path.join(root_path,"data/nyt/mid2id.json")
        },
        'gids': {
            "root": os.path.join(root_path, "data/GIDS/"),
            "train": os.path.join(root_path, "data/GIDS/train.json"),
            "test": os.path.join(root_path,"data/GIDS/test.json"),
            "rel2id": os.path.join(root_path,"data/GIDS/rel2id.json"),
            "rel2id2": os.path.join(root_path, "data/GIDS/rel2id2.json"),
            "rel2id3": os.path.join(root_path, "data/GIDS/rel2id3.json"),
            "rel_rel2": os.path.join(root_path, "data/GIDS/rel_rel2.json"),
            "rel_rel3": os.path.join(root_path, "data/GIDS/rel_rel3.json"),
            "word2vec": os.path.join(root_path,"data/GIDS/word_vec.json"),
            "entity2id": os.path.join(root_path,"data/GIDS/entity2id.json"),
            "rel_embedding": os.path.join(root_path, "data/GIDS/relation_matrix.npy"),
            "rel_embedding2": os.path.join(root_path, "data/GIDS/relation_matrix2.npy"),
            "rel_embedding3": os.path.join(root_path, "data/GIDS/relation_matrix3.npy"),
            "deepwalk": os.path.join(root_path,"data/GIDS/normalized_all.txt"),
            "entitytype": os.path.join(root_path,"data/GIDS/coarse_type_info.json"),
            "mid2id": os.path.join(root_path,"data/GIDS/mid2id.json")
        }
    }
class model:
    MODE_INSTANCE = 0
    #super parameter

    max_length = 120
    batch_size  = 160
    entity_type_embedding_size  = 15

    train_level  = MODE_RELFACT_BAG
    test_level =  MODE_ENTPAIR_BAG
    encoder = "pcnn"
    selector = "ave"

    max_epoch = 50


    learning_rate = 0.35
    optimizer = tf.train.GradientDescentOptimizer
    #optimizer = tf.train.AdamOptimizer

    pos_embedding_dim = 5

    pcnn_kernel_size = 3
    pcnn_hidden_size = 230
    pcnn_stride_size = 1

    pcnn_activiation = tf.nn.relu

    gru_hidden_size = 100
    bgwa_activiation_activiation = tf.nn.tanh
    gpu_list = [0]

    test_epoch = 1
    use_deepwalk = False

    alpha = 0.15
