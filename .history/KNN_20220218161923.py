from ast import Delete
from tkinter import END
from turtle import color
import pandas as pd
import numpy as np
import random
import math
from collections import Counter


def load_data(path, rate=0.8):

    names = ['height', 'weight', 'shoe', 'sex']
    data = pd.read_csv(path, names=names, encoding="gbk")
    data = data.drop([0])
    #print(data)
    randIndex = random.sample(range(0, len(data)), len(data))
    trainSet = data.iloc[randIndex[:int(len(data) * rate)],:]
    testSet = data.iloc[randIndex[int(len(data) * rate):],:]
    trainLabel = trainSet['sex']
    testLabel = testSet['sex']
    trainSet = trainSet.iloc[:,0:3]
    testSet = testSet.iloc[:,0:3]
    return np.array(trainSet), np.array(trainLabel), np.array(testSet), np.array(testLabel)

def load_data_forscatter():
    names = ['height', 'weight', 'shoe', 'sex']
    data = pd.read_csv('data1.csv', names=names, encoding="gbk")
    data = data.drop([0])
    man = data[data['sex'] == '男'].iloc[:, 0:4]
    famale = data[data['sex'] == '女'].iloc[:, 0:4]
    return np.array(man), np.array(famale)

def load_data_forscatter_after(trainData,trainLabel):
    for i in range(0,len(trainData)):
        man = []
        famale =[]
        if(trainLabel[i]=='男'): 
            man.append(trainData[i])
        else:
            famale.append(trainData[i])

    return np.array(man), np.array(famale)

def euclideanDistance_two_loops(train_X, test_X):

#print(train_X.shape[0])
    #print("个")
    train_X2=train_X
    i=0
    while(i<train_X2.shape[0]):
        for j in range(test_X.shape[0]):
            match=1
            for k in range(train_X2.shape[1]):
                if(train_X2[i][k]!=test_X[j][k]):
                    match=0
            if(match==1):
                train_X2=np.delete(train_X2,i,axis = 0)
                #print("match")
                i=i-1
                break
        i=i+1
    #print(train_X2.shape[0])
    #print("个\n\n")

    num_test = test_X.shape[0]
    num_train = train_X2.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = list(map(float,test_X[i]))
            train_line = list(map(float,train_X2[j]))
            temp = np.subtract(test_line, train_line)
            temp = np.power(temp, 2)
            dists[i][j] = np.sqrt(temp.sum())
    return dists




def predict_labels(dists, labels, k=1):

    num_test = dists.shape[0]
    y_pred = []
    for i in range(num_test):
        index = np.argsort(dists[i])
        index = index[:k]
        closest_y = labels[index]
        name, _ = Counter(closest_y).most_common(1)[0]
        y_pred.append(name)
    return y_pred


def getAccuracy(y_pred, y):

    num_correct = np.sum(y_pred == y)
    accuracy = float(num_correct) / len(y)
    return accuracy


def plot_k_and_accuracy(k_array, acc_array):
    max_accuracy = np.max(acc_array)
    k_max_accuracy = np.where(acc_array == max_accuracy)

    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = '12'  # 12号字体
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    plt.rcParams['font.sans-serif']=['SimHei']
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set(xlabel='K',
           ylabel='Accuracy',
           xlim=(np.min(k_array) - 5, np.max(k_array) + 5),
           ylim=(0.48, 1.02))
    ax.set_title('KNN - 准确率随K值变化情况', fontsize=16)

    ax.plot(k_array, acc_array, lw=2)
    # 标注 最高正确率 和 对应K值
    ax.vlines(x=k_array[np.max(k_max_accuracy)],
              ymin=ax.get_ylim()[0], ymax=max_accuracy,
              colors='grey', linewidth=1, linestyles='dashed')
    ax.hlines(y=max_accuracy,
              xmin=ax.get_xlim()[0], xmax=k_array[np.max(k_max_accuracy)],
              colors='grey', linewidth=1, linestyles='dashed')
    ax.text(k_array[np.max(k_max_accuracy)] + 2, max_accuracy - 0.15,
            f'Highest Accuracy = {max_accuracy:.4f}\n'
            f'when K = {k_array[k_max_accuracy]}')
    plt.show()

def plot_distrbution():
    man,famale = load_data_forscatter()
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，为了显示中文
    plt.rcParams['font.size'] = '12'  # 12号，增大字体
    fig = plt.figure()
    fig.suptitle('样本分布',fontsize = 20)  # 总标题
    x1 = man[:,0].astype(np.float64)
    y1 = man[:,1].astype(np.float64)
    z1 = man[:,2].astype(np.float64)
    x2 = famale[:,0].astype(np.float64)
    y2 = famale[:,1].astype(np.float64)
    z2 = famale[:,2].astype(np.float64)

    plt.subplot(221)#身高体重
    plt.scatter(x1, y1,color = 'blue',label='man')
    plt.scatter(x2, y2,color = 'red',label='famale')
    plt.ylabel('weight')
    plt.xlabel('height')
    plt.xlim = np.arange(140,190)
    plt.ylim = np.arange(40,90)

    plt.subplot(222)#体重鞋码
    plt.scatter(z1, y1,color = 'blue',label='man')
    plt.scatter(z2, y2,color = 'red',label='famale')
    plt.ylabel('weight')
    plt.xlabel('shoe')
    plt.xlim = np.arange(35,45)
    plt.ylim = np.arange(40,90)
    plt.legend(loc='right', bbox_to_anchor=(1.2, 1.12),fontsize = 14)

    plt.subplot(223)#身高鞋码
    plt.scatter(z1, x1,color = 'blue',label='man')
    plt.scatter(z2, x2,color = 'red',label='famale')
    plt.xlabel('height')
    plt.ylabel('shoe')
    plt.ylim = np.arange(35,45)
    plt.xlim = np.arange(140,190)
    plt.show()

def plot_distrbution_after(trainData,trainLabel):
    man,famale = load_data_forscatter(trainData,trainLabel)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，为了显示中文
    plt.rcParams['font.size'] = '12'  # 12号，增大字体
    fig = plt.figure()
    fig.suptitle('样本分布',fontsize = 20)  # 总标题
    x1 = man[:,0].astype(np.float64)
    y1 = man[:,1].astype(np.float64)
    z1 = man[:,2].astype(np.float64)
    x2 = famale[:,0].astype(np.float64)
    y2 = famale[:,1].astype(np.float64)
    z2 = famale[:,2].astype(np.float64)

    plt.subplot(221)#身高体重
    plt.scatter(x1, y1,color = 'blue',label='man')
    plt.scatter(x2, y2,color = 'red',label='famale')
    plt.ylabel('weight')
    plt.xlabel('height')
    plt.xlim = np.arange(140,190)
    plt.ylim = np.arange(40,90)

    plt.subplot(222)#体重鞋码
    plt.scatter(z1, y1,color = 'blue',label='man')
    plt.scatter(z2, y2,color = 'red',label='famale')
    plt.ylabel('weight')
    plt.xlabel('shoe')
    plt.xlim = np.arange(35,45)
    plt.ylim = np.arange(40,90)
    plt.legend(loc='right', bbox_to_anchor=(1.2, 1.12),fontsize = 14)

    plt.subplot(223)#身高鞋码
    plt.scatter(z1, x1,color = 'blue',label='man')
    plt.scatter(z2, x2,color = 'red',label='famale')
    plt.xlabel('height')
    plt.ylabel('shoe')
    plt.ylim = np.arange(35,45)
    plt.xlim = np.arange(140,190)
    plt.show()

def clipping_KNN(membernum):
    trainData, trainLabel, testData, testLabel = load_data("data1.csv", 0.67)
    New_traindata = []#存放剪辑后的新训练数据集

    num = np.shape(trainData)[0]
    groupnum = math.ceil( num / membernum)

    for i in range(0,groupnum):
        trainData_cut = np.array_split(trainData,groupnum)[i] #一个数据集切片
        trainLabel_cut= np.array_split(trainLabel,groupnum)[i]
       
        dists = euclideanDistance_two_loops(trainData, trainData_cut)
        y_pred = predict_labels(dists, trainLabel, k=5)
        deletelist = []         #用于确定要删除的数据的index
        for j in range(0,len(trainLabel_cut)):
            if (y_pred[j] != trainLabel_cut[j]):
                deletelist.append(j)
        trainData_cut = np.delete(trainData_cut,deletelist,axis=0)
        trainLabel_cut= np.delete(trainLabel_cut,deletelist,axis=0)
        for n in range(len(trainData_cut)):
            New_traindata.append(trainData_cut[n])
            New_trainlabel.append(trainLabel_cut[n])

    New_traindata = np.array(New_traindata)
    return New_traindata,New_trainlabel
    
if __name__ == "__main__":
    trainData, trainLabel, testData, testLabel = load_data("data1.csv", 0.67)
    dists = euclideanDistance_two_loops(trainData, testData)
    """
    for i in range(3, 150, 2):
        y_pred = predict_labels(dists, trainLabel, k=i)
        accuracy = getAccuracy(y_pred, testLabel)
        print("K值为"+str(i)+",准确率为： "+str(accuracy))
    """

    k_array = np.arange(3, 150, 2)
    acc_array = []
    #for k in k_array:
        #y_pred = predict_labels(dists, trainLabel, k=k)
        #accuracy = getAccuracy(y_pred, testLabel)
        #acc_array.append(accuracy)
        #print("K值为" + str(k) + ",准确率为： " + str(accuracy))

    #plot_k_and_accuracy(k_array, acc_array)
    #plot_distrbution(trainData)

    num = np.shape(trainData)[0]
    groupnum = math.ceil( num / 5)
    New_trainlabel = []
    New_traindata = []
    Deleteflag = np.zeros((num,1))
    c = 0
    for i in range(0,groupnum):       
        trainData_cut = np.array_split(trainData,groupnum)[i]
        trainLabel_cut= np.array_split(trainLabel,groupnum)[i]

        dists = euclideanDistance_two_loops(trainData, trainData_cut)
        y_pred = predict_labels(dists, trainLabel, k=5)
        deletelist = []
        j = 0
        for j in range(0):
            if (y_pred[j] != trainLabel_cut[j]):
                deletelist.append(j)
                #Deleteflag[c] = 1
            j+=1
            #c += 1
        #print(deletelist)
        trainData_cut = np.delete(trainData_cut,deletelist,axis=0)
        trainLabel_cut= np.delete(trainLabel_cut,deletelist,axis=0)

        for n in range(len(trainData_cut)):
            New_traindata.append(trainData_cut[n])
            New_trainlabel.append(trainLabel_cut[n])
        c += 1
    New_traindata = np.array(New_traindata)

    #plot_distrbution(New_traindata)
    #plot_distrbution(trainData)
    #print(New_traindata.size)
    #print(np.size(New_trainlabel))
    print(c)
