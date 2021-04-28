import module
import os
import sys
import config
import tensorflow as tf
import numpy as np
import json
import matplotlib
import model_struct
import time
matplotlib.use('agg')

dataset = "agriculture"
if(len(sys.argv) > 1):
    dataset = sys.argv[1]


dataset_dir = config.dir.dataset_dir[dataset]['root']

if(not os.path.isdir(dataset_dir)):
    raise Exception("[ERROR] Dataset dir %s doesn't exit!" % dataset_dir)




start_time = time.time()
train_loader = module.data_loader.json_file_data_loader(config.dir.dataset_dir[dataset]['train'],
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
                                                        mode = config.model.train_level,
                                                        shuffle = True)

test_loader = module.data_loader.json_file_data_loader(config.dir.dataset_dir[dataset]['test'],
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
                                                       mode = config.model.test_level,
                                                       shuffle =False)



# framework = module.binary_train_framework.re_framework(test_loader,train_loader)
# test_framework = module.binary_test_framework.model_test(test_loader)
encoder = config.model.encoder
selector = config.model.selector

entity_inf = 'before'
if len(sys.argv) > 2:
    encoder = sys.argv[2]
if len(sys.argv) > 3:
    selector = sys.argv[3]
if len(sys.argv) > 4:
    entity_inf = sys.argv[4]


cur_model = model_struct.model
cur_model(train_loader,config.model.batch_size,config.model.max_length,encoder,selector,entity_inf,dataset)




model_name = dataset + "_"+encoder+"_"+selector+"_"+entity_inf

framework = module.train_framework.re_framework(dataset,test_loader,train_loader)
test_framework = module.test_framework.model_test(test_loader,cur_model,encoder,selector,entity_inf,dataset)
framework.train(cur_model,encoder,selector,entity_inf,model_name = model_name,ckpt_dir = "checkpoint",max_epoch = config.model.max_epoch,gpu_list = config.model.gpu_list)

#output result
num_params = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    local_params = 1
    for dim in shape:
        local_params *= dim.value
    num_params += local_params
print("parameters number is %d" % num_params)

auc,prec,recall = test_framework.test(model_name,test_loader,ckpt= "./checkpoint/"+model_name,return_result=True,mode=config.model.test_level)

x = np.load("./test_result/"+model_name+"_x.npy")
y = np.load("./test_result/"+model_name+"_y.npy")
### max f1
f1 = (2*(x*y)/(x + y + 1e-20)).max()
print(model_name+" auc= %f" %auc)
print(model_name+" max f1= %f" %f1)
print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300], (y[100] + y[200] + y[300]) / 3))

end_time = time.time()
print("time consume %s"%(str(end_time-start_time)))



