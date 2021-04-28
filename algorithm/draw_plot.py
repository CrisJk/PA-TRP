import numpy as np
import sys
import sklearn
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

models = sys.argv[1:]
cnt = 0
maker = ['<','|','^','s','d','x','p','*','o']

#color = ['b','g','c','#FF00FF','y','m','#FF0000']
for model_name in models:
    x = np.load("./test_result/gids/"+model_name+"_x.npy")
    y = np.load("./test_result/gids/"+model_name+"_y.npy")

    index = np.argmax((2 * x * y) / (x + y + 1e-20))
    recall = x[index]
    prec = y[index]

    # new_x = []
    # new_y = []
    #
    # now = 0.0
    # ccnt = 0
    # for i in range(x.shape[0]):
    #     if x[i]>0.6 and x[i]<0.8  and ccnt<5:
    #         ccnt+=1
    #         continue
    #     elif x[i]>0.8 and x[i]-now < 0.0001 and ccnt<500:
    #         ccnt+=1
    #         continue
    #     now = x[i]
    #     ccnt = 0
    #     new_x.append(x[i])
    #     new_y.append(y[i])
    #
    # x = np.array(new_x)
    # y = np.array(new_y)
    # print("shape ",x.shape)
    index = np.argmax((2 * x * y) / (x + y + 1e-20))
    print("index",index)
    recall = x[index]
    prec = y[index]

    f1 = (2 * x * y / (x + y + 1e-20)).max()

    makevery =1000

    #plt.plot(x, y,color[cnt],lw=2, label=model_name,linewidth=1,marker=maker[cnt],markevery=200,markersize=6)
    # if cnt == 3:
    #     plt.plot(x, y, '#F08080',lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=4)
    # elif cnt == 4:
    #     plt.plot(x, y, '#BDB76B', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=4)
    # elif cnt == 5:
    #     plt.plot(x, y, '#7B68EE', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=4)
    # elif cnt == 6:
    #     plt.plot(x, y, '#00CED1', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=6)
    # elif cnt == 7:
    #     plt.plot(x, y, '#FFDD33', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery,
    #              markersize=4)
    # elif cnt == 8:
    #     plt.plot(x, y, '#E3321A',lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=4)
    #
    # else:
    #     plt.plot(x, y, lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=makevery, markersize=4)

    if cnt==0:
        plt.plot(x, y, '#F08080',lw=2,label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)
    elif cnt == 1:
        plt.plot(x, y, '#BDB76B', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)
    elif cnt == 2:
        plt.plot(x, y, '#7B68EE', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)
    # elif cnt == 3:
    #     plt.plot(x, y, '#00CED1', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)
    elif cnt == 5:
        plt.plot(x, y, '#E3321A', lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)
    else:
        plt.plot(x, y, lw=2, label=model_name, linewidth=1, marker=maker[cnt], markevery=1000, markersize=6)


    auc = metrics.auc(x=x, y=y)

    print(model_name+" auc= %f" %auc)
    print(model_name+" max f1= %f pre=%f recall=%f" %(f1,prec,recall))

    #y = sorted(y,reverse=True)
    first = 5521
    print(first)


    print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[99], y[199], y[299], (y[99] + y[199] + y[299]) / 3))

    plt.xlabel("Recall",fontsize=12)
    plt.ylabel("Precision",fontsize=12)
    # plt.xlim([0,max(1.0,np.max(y)+0.2)])
    # plt.ylim([np.min(x),max(np.max(x)+0.2,1.0)])
    # plt.ylim([0.0, 1.0])
    # plt.xlim([0.0, 1.0])
    plt.ylim([0.7, 1.0])
    plt.xlim([0.0, 1.0])
    cnt+=1

#plt.title("Precision-Recall")
plt.legend(loc = "upper right")
plt.grid(True)

plt.savefig("./test_result/"+'2_pr_curve',dpi=350)

