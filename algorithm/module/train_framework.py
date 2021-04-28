# -*- coding:utf-8 -*-
import config
import tensorflow as tf
import os
import time
import numpy as np
import sys
from . import test_framework

import tensorflow.contrib.slim as slim


class re_framework:
    #bag level
    MODE_BAG = 0
    #Instance level
    MODE_INS = 1

    def __init__(self,dataset,test_data_loader,train_data_loader=None):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None
        self.dataset = dataset
    def __iter__(self):
        return self

    def average_gradients(self,tower_grads):
        '''
        多GPU,用于计算平均梯度
        :param tower_grads:tower_grads为一个列表，列表中的每一项都是一个tuple列表

        [ [(grad0,var0),(grad1,var1),...],[(grad0,var0),(grad1,var1),...] ]

        :return: 返回平均梯度，((avg_grad0,var0),(avg_grad1,var1),...)
        '''
        average_grads = []

        '''
        zip(*tower_grads)
        假设tower_grads = [[(1,0),(2,1),(3,2)],[(4,0),(5,1),(6,2)],[(7,0),(8,1),(9,2)]]
        zip(*tower_grads) = 
        [((1, 0), (4, 0), (7, 0)),
        ((2, 1), (5, 1), (8, 1)),
        ((3, 2), (6, 2), (9, 2))]
           
        '''
        for grad_and_vars in zip(*tower_grads):
            grads = []
            # 每个grad_and_vars 形如 ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN)) , ((grad1_gpu0,var1_gpu0),...,(grad1_gpuN,var1_gpuN))
            for g,_ in grad_and_vars:
                expand_g = tf.expand_dims(g,0)
                grads.append(expand_g)

            grad = tf.concat(grads,0)
            grad = tf.reduce_mean(grad,0)

            v= grad_and_vars[0][1]
            average_grads.append((grad,v))
        return average_grads

    def one_step_multi_models(self,models,batch_data_gen,run_array,return_label=True):
        feed_dict = {}
        batch_label = []
        for model in models:
            batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size // len(models))
            feed_dict.update({
                model.word: batch_data['word'],
                model.pos1: batch_data['pos1'],
                model.pos2: batch_data['pos2'],
                model.label: batch_data['rel'],
                model.label2: batch_data['rel2'],
                model.label3: batch_data['rel3'],
                model.ins_label: batch_data['ins_rel'],
                model.ins_label2: batch_data['ins_rel2'],
                model.ins_label3: batch_data['ins_rel3'],
                model.scope: batch_data['scope'],
                model.length: batch_data['length'],
                model.entity1: batch_data['entity1'],
                model.entity2: batch_data['entity2'],
                model.entity1_type: batch_data['entity1_type'],
                model.entity2_type: batch_data['entity2_type'],
                model.entity1_type_mask: batch_data['entity1_type_mask'],
                model.entity2_type_mask: batch_data['entity2_type_mask'],
                model.entity1_type_length: batch_data['entity1_type_length'],
                model.entity2_type_length: batch_data['entity2_type_length']
                #model.bc_embedding: batch_data['bc']
            })
            if 'mask' in batch_data and hasattr(model,"mask"):
                feed_dict.update({
                    model.mask: batch_data['mask']
                })
            batch_label.append(batch_data['rel'])

        result = self.sess.run(run_array,feed_dict)

        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result,feed_dict


    def train(self,model,encoder,selector,entity_inf,model_name,ckpt_dir = './checkpoint',summary_dir ='./summary',test_result_dir='./test_result',
              learning_rate = config.model.learning_rate,max_epoch = config.model.max_epoch,pretrain_model = None
              ,test_epoch = config.model.test_epoch ,optimizer = config.model.optimizer,gpu_list= config.model.gpu_list):

        gpu_nums = len(gpu_list)
        assert(self.train_data_loader.batch_size % gpu_nums == 0)
        print("start training...")

        #Init
        configure = tf.ConfigProto()
        configure.allow_soft_placement = True
        configure.gpu_options.allow_growth = True
        configure.gpu_options.per_process_gpu_memory_fraction = 0.6

        learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)

        optimizer1 = optimizer(learning_rate)
        tower_grads= []
        tower_models = []
        gpu_list = config.model.gpu_list

        self.sess = tf.Session(config=configure)
        path = os.path.join(ckpt_dir, model_name)

        for gpu_id in gpu_list:
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):

                    cur_model = model(self.train_data_loader,config.model.batch_size // gpu_nums,self.train_data_loader.max_length,encoder,selector,entity_inf,self.dataset)
                    tower_grads.append(optimizer1.compute_gradients(cur_model.loss()))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss",cur_model.loss())
                    tf.add_to_collection("train_logit",cur_model.train_logit())

        interval = tower_models[0].interval()
        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)

        train_logit_collection = tf.get_collection("train_logit")
        train_logit = tf.concat(train_logit_collection,0)

        grads = self.average_gradients(tower_grads)

        train_op = optimizer1.apply_gradients(grads)

        #Saver
        saver = tf.train.Saver(max_to_keep=None,save_relative_paths=True)

        self.sess.run(tf.global_variables_initializer())
        #Training
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0

        test = test_framework.model_test(self.test_data_loader,model,encoder,selector,entity_inf,self.dataset)
        sstep = 0
        with open('loss.txt','w',encoding='utf8') as fw:
            for epoch in range(max_epoch):
                print('###### Epoch' + str(epoch) + '######')
                tot_correct = 0
                tot_not_na_correct = 0
                tot = 0
                tot_not_na = 0
                i = 0
                time_sum = 0
                beta = 0
                alpha = 0
                gamma = 0


                iter_interval = None
                while True:
                    time_start = time.time()
                    try:
                        (iter_loss, iter_logit,_train_op,iter_interval,iter_label),feed_dict= self.one_step_multi_models(tower_models,self.train_data_loader,[loss,train_logit,train_op,interval])
                    except StopIteration:
                        break
                    sstep+=1
                    alpha,beta = iter_interval

                    time_end = time.time()

                    t = time_end - time_start

                    time_sum += t
                    iter_output = iter_logit.argmax(-1)
                    iter_correct = (iter_output == iter_label).sum()
                    iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                    tot_correct += iter_correct
                    tot_not_na_correct += iter_not_na_correct
                    tot += iter_label.shape[0]
                    tot_not_na += (iter_label != 0).sum()
                    fw.write(str(iter_loss)+'\n')
                    if tot_not_na > 0:
                        sys.stdout.write(
                            "epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (
                            epoch, i, t,iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                        sys.stdout.flush()
                    i += 1


                print(iter_interval)
                print("not NA accuracy: %f, accuracy: %f\r" %
                      (float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))

                print("\nAverage iteration time: %f" % (time_sum / i))
                print("alpha %f , beta %f, %f"%( alpha, beta,gamma) )
                if (epoch+1)%test_epoch== 0:

                    metric,cur_prec,cur_recall = test.test(model_name,self.test_data_loader,ckpt=None,return_result=False,mode=config.model.test_level,sess = self.sess)
                    x = np.array(cur_recall)
                    y = np.array(cur_prec)
                    f1 = ((2 * x * y) / (x + y + 1e-20)).max()
                    index = np.argmax((2 * x * y) / (x + y + 1e-20))
                    recall = x[index]
                    prec = y[index]
                    print("Prec: ",prec)
                    print("Recall: ",recall)
                    print("F1: ",f1)
                    if metric > best_metric:
                        best_metric = metric
                        best_prec = cur_prec
                        best_recall = cur_recall
                        print("Best Model, storing...")
                        if not os.path.isdir(ckpt_dir):
                            os.mkdir(ckpt_dir)
                        path = saver.save(self.sess,os.path.join(ckpt_dir,model_name))
                        print("Finish Storing")
                        not_best_count = 0
                    else:
                        not_best_count += 1
                    if not_best_count >=100:
                        break


            print("######")
            print("Finish training " + model_name)
            print("Best epoch auc = %f" %(best_metric))


            if (not best_prec is None) and (not best_recall is None):
                if not os.path.isdir(test_result_dir):
                    os.mkdir(test_result_dir)

                np.save(os.path.join(test_result_dir,model_name+'_x.npy'),best_recall)
                np.save(os.path.join(test_result_dir,model_name+'_y.npy'),best_prec)



