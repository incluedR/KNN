import pandas as pd
import numpy as np
import random
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

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
    man = data[data['sex'] == '男'].iloc[:, 1:4]
    famale = data[data['sex'] == '女'].iloc[:, 2:4]
    return man, famale



def euclideanDistance_two_loops(train_X, test_X):

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = list(map(float,test_X[i]))
            train_line = list(map(float,train_X[j]))
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

def plot_distrbution(trainData):
    man,famale = load_data_forscatter()
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.scatter(x1, y1, z1)
    
    print(man)

    #ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    #ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    #ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    #plt.show()




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
    for k in k_array:
        y_pred = predict_labels(dists, trainLabel, k=k)
        accuracy = getAccuracy(y_pred, testLabel)
        acc_array.append(accuracy)
        #print("K值为" + str(k) + ",准确率为： " + str(accuracy))

    #plot_k_and_accuracy(k_array, acc_array)
    plot_distrbution(trainData)
   