# -*- coding: utf-8 -*-
import config
import tensorflow as tf
import numpy as np
import sklearn.metrics
import sys
import os
import json
class model_test:
    #Instance level
    MODE_INS = 0
    # bag level
    MODE_BAG = 1
    test_result_dir = os.path.join(os.getcwd(),"offline_test_result")

    def __init__(self, test_data_loader,model,encoder,selector,entity_inf,dataset):
        self.test_data_loader = test_data_loader
        dataset_name = test_data_loader.file_name.split('/')[-2]
        if dataset_name == 'nyt':
            with open('data/nyt/entity2id.json','r',encoding='utf8') as fr:
                e2id = json.load(fr)
            self.id2entity = {}
            for item in e2id:
                self.id2entity[e2id[item]] = item

        if dataset_name == 'GIDS':
            with open('data/GIDS/entity2id.json','r',encoding='utf8') as fr:
                e2id = json.load(fr)
            self.id2entity = {}
            for item in e2id:
                self.id2entity[e2id[item]] = item

        self.id2word = {}
        with open('_processed_data/'+dataset_name+'_word_vec_id2word.json', 'r', encoding='utf8') as fr:
            self.id2word = json.load(fr)
        self.model = model(test_data_loader, test_data_loader.batch_size, test_data_loader.max_length, encoder, selector,entity_inf,dataset)
    def one_step(self, model, batch_data, sess,run_array):
        #print(batch_data['sentence'])
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
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
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        ret_result = sess.run(run_array, feed_dict)
        return ret_result

    def test(self, model_name,test_data_loader,ckpt=None, return_result=True, mode=MODE_BAG,sess= None,save_result = False,training = True,show_att=False):
        return self.__test_bag__(model_name,test_data_loader,mode,ckpt,return_result,sess,save_result,training,show_att)


    def __test_bag__(self,model_name,test_data_loader,mode,ckpt=None, return_result=False,sess = None,save_result=False,training=True,show_att=True):
        print("Testing...")
        gpu_id = config.model.gpu_list[0]

        with tf.device("/gpu:%d" % gpu_id):

            if sess == None:
                configure = tf.ConfigProto()
                configure.allow_soft_placement = True
                configure.gpu_options.allow_growth = True
                configure.gpu_options.per_process_gpu_memory_fraction = 1.0
                sess = tf.Session(config=configure)
            if not ckpt is None:
                saver = tf.train.Saver(save_relative_paths=True)
                saver.restore(sess, ckpt)
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            entpair_tot = 0
            test_result = []
            na_test_result = []
            pred_result = []

            loss = self.model.test_loss()



            import time
            for i, batch_data in enumerate(test_data_loader):
                if show_att:
                    cur_loss, iter_logit,interval = self.one_step(self.model, batch_data,sess,[loss, self.model.test_logit(),self.model.interval()])


                    print(interval)

                else:
                    cur_loss, iter_logit = self.one_step(self.model, batch_data, sess,
                                                                   [loss, self.model.test_logit()])

                #print(iter_logit)
                iter_output = iter_logit.argmax(-1)
                iter_correct = (iter_output == batch_data['rel']).sum()
                iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += batch_data['rel'].shape[0]
                tot_not_na += (batch_data['rel'] != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("[TEST] step %d | na %d,nat %d,all %d, allt %d not NA accuracy: %f, accuracy: %f\r" % (
                        i, tot_not_na,tot_not_na_correct,tot,tot_correct, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()

                for idx in range(0,len(iter_logit)):

                    for rel in range(1, test_data_loader.rel_tot):
                        idx0 = batch_data['idx'][idx][0]
                        idx1 = batch_data['idx'][idx][1]
                        if idx0 <= idx1:
                            if mode == model_test.MODE_BAG:
                                positive = 0
                                if batch_data['multi_rel'][idx][0] == 0:
                                    positive = 1
                                test_result.append({'score': float(iter_logit[idx][rel]), 'flag': int(batch_data['multi_rel'][idx][rel]),
                                                    'entity1': int(batch_data['entity1'][batch_data['scope'][idx][0]]),
                                                    'entity2': int(batch_data['entity2'][batch_data['scope'][idx][0]]),
                                                    #'sentence': self.model.data_loader.data_sentence[idx0:idx1],
                                                    'gt_rel': int(batch_data['rel'][idx]), 'rel': rel,
                                                    #'words':batch_data['word'][batch_data['scope'][idx][0]].tolist(),
                                                    'multi_rel':batch_data['multi_rel'][idx].tolist(),
                                                    #'positive':positive,'na_true': int(batch_data['multi_rel'][idx][0])
                                })

                            else:
                                test_result.append(
                                    {'score': iter_logit[idx][rel], 'flag': batch_data['rel'][idx] == rel,
                                     'entity1': batch_data['entity1'][idx],
                                     'entity2': batch_data['entity2'][idx],
                                     'sentence': self.model.data_loader.data_sentence[idx0:idx1],
                                     'gt_rel': batch_data['rel'][idx], 'rel': rel})
                                entpair_tot += 1



            sorted_test_result = sorted(test_result, key=lambda x: x['score'])




            prec = []
            recall = []
            correct = 0

            for i, item in enumerate(sorted_test_result[::-1]):
                correct += item['flag']
                prec.append(float(correct) / (i + 1))
                recall.append(float(correct) / test_data_loader.relfact_tot)

            auc = sklearn.metrics.auc(x=recall, y=prec)
            print("\n[TEST] auc: {}".format(auc))
            print("Finish testing")

            if save_result == True:
                if not os.path.exists(model_test.test_result_dir):
                    os.mkdir(model_test.test_result_dir)
                np.save(os.path.join(model_test.test_result_dir, model_name + '_x.npy'), recall)
                np.save(os.path.join(model_test.test_result_dir, model_name + '_y.npy'), prec)

            if not training:
                sess.close()


            print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(prec[99], prec[199], prec[299],
                                                                            (prec[99] + prec[199] + prec[
                                                                                299]) / 3))
            return auc,prec,recall

