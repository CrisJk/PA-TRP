# -*- coding:utf-8 -*-



from matplotlib import pyplot as plt
import json
from matplotlib.pyplot import MultipleLocator

from tqdm import tqdm
def draw_plot(train,test,co_range,dataset = 'NYT'):



    e1_e2 = {}
    for item in tqdm(train):
        e1_id = item['head']['word'].lower()
        e2_id = item['tail']['word'].lower()
        if e1_e2.get(e1_id+'#'+e2_id) is None:
            e1_e2[e1_id+'#'+e2_id] = 0

        e1_e2[e1_id+'#'+e2_id] += 1


    for item in tqdm(test):
        e1_id = item['head']['word'].lower()
        e2_id = item['tail']['word'].lower()
        if e1_e2.get(e1_id + '#'+e2_id) is None:
            e1_e2[e1_id+'#'+e2_id] = 0

        e1_e2[e1_id+'#'+e2_id] += 1

    reverse = {}

    for key in e1_e2:
        if reverse.get(e1_e2[key]) is None:
            reverse[e1_e2[key]]= 0
        reverse[e1_e2[key]] += 1

    range_reverse = {}
    for key in reverse:
        for x in co_range:
            if key >= x[0] and key <= x[1]:

                if range_reverse.get(x) is None:
                    range_reverse[x] = reverse[key]
                else:
                    range_reverse[x] += reverse[key]





    x = []
    y = []


    for key in reverse:

        x.append(int(key))
        y.append(reverse[int(key)])

    reverse_list = sorted(reverse.items(),key=lambda x:x[0])


    xx= []
    yy =[]

    r = co_range[0][1]- co_range[0][0]
    for key in co_range:
        xx.append(key[1])
        # else:
        #     xx.append('>'+str(int(key[0]+r/2)))
        if range_reverse.get(key) is None:
            range_reverse[key] = 0
        yy.append(range_reverse[key])

    return xx,yy


###########################################################################
#draw NYT
###########################################################################

with open('data/nyt/train-reading-friendly.json','r',encoding='utf8') as fr:
    train = json.load(fr)

with open('data/nyt/test.json','r',encoding='utf8') as fr:
    test = json.load(fr)
#
#co_range = [(0, 10), (10, 20), (20, 30), (30, 40),(40,50),(50,60),(60,70),(70,80),(80,100000000000)]
co_range = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),(50,10000000)]
x,y = draw_plot(train,test,co_range,'NYT')

import numpy as np
entity_sum = np.sum(np.array(y))

now = 0
for i in range(len(y)):
    now+=y[i]
    if(1.0*now/entity_sum)>0.8:
        print(i)
        break

print(x)
print(y)
#plt.bar(x,y)
plt.plot(x,y)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim([0,51])

x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.xlabel('Co-occurrence frequencies',fontsize=14)
plt.ylabel('Entity pairs number',fontsize=14)
#plt.title('distribution of entity pairs co-occurrence on NYT dataset')

#plt.yscale("log")
plt.subplot(111)
plt.tight_layout()
plt.savefig('nyt_statistic.png',dpi=300)
plt.close()

###########################################################################
#draw GDS
###########################################################################


with open('data/GIDS/train.json','r',encoding='utf8') as fr:
    gds_train = json.load(fr)

with open('data/GIDS/test.json','r',encoding='utf8') as fr:
    gds_test = json.load(fr)



co_range = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
            (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 1000000)]
x,y  = draw_plot(gds_train,gds_test,co_range,'GDS')

print(x)
print(y)

import numpy as np
entity_sum = np.sum(np.array(y))

now = 0
for i in range(len(y)):
    now+=y[i]
    if(1.0*now/entity_sum)>0.8:
        print(i)
        break

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim([0,19])
plt.plot(x,y)


x_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.xlabel('Co-occurrence frequencies',fontsize=14)
plt.ylabel('Entity pairs number',fontsize=14)

#plt.ylim(0.0,500000)
#plt.title('distribution of entity pairs co-occurrence on GDS dataset')
# plt.yscale("log")
plt.subplot(111)
plt.tight_layout()
plt.savefig('gds_statistic.png',bbox_inches = 'tight',dpi=300)
