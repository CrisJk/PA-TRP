# -*- coding: utf8 -*-

#### test via pretrained module
import module
import config
import sys
import model_struct
import numpy as np
import os
dataset = "agriculture"
model_name = config.model.encoder+"_"+config.model.selector
if len(sys.argv) > 1:
    dataset = sys.argv[1]

if len(sys.argv) > 2:
    model_name = sys.argv[2]


encoder = model_name.split("_")[-3]
selector = model_name.split("_")[-2]
entity_inf = model_name.split("_")[-1]


# model_name = dataset+"_"+model_name
test_result_dir = os.path.join(os.getcwd(),"offline_test_result")
first = 5521//10

def print_result(auc,prec,recall,f1,y):
    print("%s auc is %f" % (model_name, auc))
    # P R F 取使f值最大的阈值

    print("Precision %f, Recall %f, F1 %f" % (prec, recall, f1))


    #p@N bag
    print('P@N bag    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[99], y[199], y[299],
                                                                   (y[99] + y[199] + y[299]) / 3))



def evaluation(test_data_loader):
    cur_model = model_struct.model
    cur_model(test_data_loader, config.model.batch_size, config.model.max_length,encoder,selector,entity_inf,dataset)
    framework = module.test_framework
    framework = framework.model_test(test_data_loader,cur_model,encoder,selector,entity_inf,dataset)


    auc, prec, recall = framework.test(model_name, test_data_loader, ckpt="./checkpoint/" + model_name,
                                       return_result=True, mode=config.model.test_level, sess=None, save_result=True,training = False,show_att=False)
    x = np.array(recall)
    y = np.array(prec)
    f1 = ((2 * x * y) / (x + y + 1e-20)).max()
    index = np.argmax((2 * x * y) / (x + y + 1e-20))

    recall = x[index]
    prec = y[index]

    if not os.path.isdir(test_result_dir):
        os.mkdir(test_result_dir)

    np.save(os.path.join(test_result_dir, model_name + '_x.npy'), x)
    np.save(os.path.join(test_result_dir, model_name + '_y.npy'), y)

    return  auc,prec, recall,f1,y




test_data_loader_all = module.data_loader.json_file_data_loader(config.dir.dataset_dir[dataset]['test'],
                                                            config.dir.dataset_dir[dataset]['word2vec'],
                                                            config.dir.dataset_dir[dataset]['rel2id'],
                                                            config.dir.dataset_dir[dataset]['rel2id2'],
                                                            config.dir.dataset_dir[dataset]['rel2id3'],
                                                            config.dir.dataset_dir[dataset]['deepwalk'],
                                                            config.dir.dataset_dir[dataset]['rel_rel2'],
                                                            config.dir.dataset_dir[dataset]['rel_rel3'],
                                                            config.dir.dataset_dir[dataset]['entity2id'],
                                                            config.dir.dataset_dir[dataset]['mid2id'],
                                                            config.dir.dataset_dir[dataset]['entitytype'],
                                                            mode=config.model.test_level,
                                                            shuffle = False,sen_num_bag='all')


# p_one_100 = 0
# p_one_200 = 0
# p_one_300 = 0
# one_prec = 0
# one_recall = 0
# one_f1 = 0
# p_two_100 = 0
# p_two_200 = 0
# p_two_300 = 0
# two_prec =  0
# two_recall = 0
# two_f1 = 0
# times = 5
#
#
# for i in range(times):
#     test_data_loader_one = module.data_loader.json_file_data_loader(config.dir.dataset_dir[dataset]['test'],
#                                                                     config.dir.dataset_dir[dataset]['word2vec'],
#                                                                     config.dir.dataset_dir[dataset]['rel2id'],
#                                                                     config.dir.dataset_dir[dataset]['coocurrence'],
#                                                                     config.dir.dataset_dir[dataset]['deepwalk'],
#                                                                     config.dir.dataset_dir[dataset]['entity2id'],
#                                                                     config.dir.dataset_dir[dataset]['mid2id'],
#                                                                     config.dir.dataset_dir[dataset]['entitytype'],
#                                                                     mode=config.model.test_level,
#                                                                     shuffle=False, sen_num_bag='one')
#     auc,prec,recall,f1,y = evaluation(test_data_loader_one)
#     p_one_100 += y[99]
#     p_one_200 += y[199]
#     p_one_300 += y[299]
#     one_prec += prec
#     one_recall += recall
#     one_f1 += f1
#
#
#
#
#
# for i in range(times):
#     test_data_loader_two = module.data_loader.json_file_data_loader(config.dir.dataset_dir[dataset]['test'],
#                                                                     config.dir.dataset_dir[dataset]['word2vec'],
#                                                                     config.dir.dataset_dir[dataset]['rel2id'],
#                                                                     config.dir.dataset_dir[dataset]['coocurrence'],
#                                                                     config.dir.dataset_dir[dataset]['deepwalk'],
#                                                                     config.dir.dataset_dir[dataset]['entity2id'],
#                                                                     config.dir.dataset_dir[dataset]['mid2id'],
#                                                                     config.dir.dataset_dir[dataset]['entitytype'],
#                                                                     mode=config.model.test_level,
#                                                                     shuffle=False, sen_num_bag='two')
#     auc,prec,recall,f1,y= evaluation(test_data_loader_two)
#     p_two_100 += y[99]
#     p_two_200 += y[199]
#     p_two_300 += y[299]
#     two_prec += prec
#     two_recall += recall
#     two_f1 += f1
#
#
#
# print("Precision %f, Recall %f, F1 %f" % (one_prec/times, one_recall/times, one_f1/times))
#
# # p@N bag
# print('P@N bag one    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(p_one_100/times, p_one_200/times, p_one_300/times,
#
#                                                                 (p_one_100 + p_one_200 + p_one_300) / (3*times)))
# print("Precision %f, Recall %f, F1 %f" % (two_prec/times, two_recall/times, two_f1/times))
#
# # p@N bag
# print('P@N bag two   P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(p_two_100/times, p_two_200/times, p_two_300/times,
#                                                                 (p_two_100 + p_two_200 + p_two_300) / (3*times)))

auc,prec,recall,f1,y= evaluation(test_data_loader_all)
print_result(auc,prec,recall,f1,y)










