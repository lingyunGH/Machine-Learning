import numpy as np
import copy
from sklearn.datasets import make_blobs
import sys

def randCent(data,k):
    n_feature = data.shape[1]
    '''生成k个随机中心'''
    centroids = np.zeros((k,n_feature))
    for i in range(n_feature):
        min_j = np.min(data[:,i])
        ranges = np.max(data[:,i])-min_j
        for k_ in range(k):  ##k-means对初始值要求很高，为了更快的收敛，初始值最好设置的够均衡
            centroids[k_,i] = min_j+ranges*np.random.rand()
    return centroids


def caculate(data,center):
    '''计算两个向量之间的距离'''
    return np.sqrt(np.sum(np.power(data-center,2)))

def kmeans(data,k):
    ###随机选择质心
    centroids = randCent(data,k)
    data_label = np.zeros((data.shape[0],1))
    pre = np.zeros((k,data.shape[1]))
    erro = caculate(centroids, pre)
    while erro>0:   ##当质心不再改变的时候模型收敛
        ###计算所有点到各个质心的距离
        for i in range(data.shape[0]):
            min_distance = sys.maxsize
            for k_ in range(k):
                distance = caculate(data[i,:],centroids[k_,:])
                if distance<min_distance:
                    min_distance = distance
                    min_index = k_
            data_label[i] = min_index
        pre =copy.deepcopy(centroids)   ##保存上一次的质心
        ##重新计算质心
        for k_ in range(k):
            cur_index = np.where(data_label == k_)
            centroids[k_,:] = np.mean(data[cur_index[0],:],axis=0)  ##计算每个类别下的样本的均值
        erro = caculate(centroids,pre)  ##计算新的质心和上一次质心之间的变化，

    return centroids,data_label

def show(dataSet, k, centroids, clusterAssment):
     from matplotlib import pyplot as plt
     plt.figure(figsize=(12, 12))
     numSamples, dim = dataSet.shape
     mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
     for i in range(numSamples):
         markIndex = int(clusterAssment[i, 0])
         plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
     mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
     for i in range(k):
         plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
     plt.show()


if __name__=="__main__":
    n_samples = 1000
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    K = 3  ##类别数目
    centroids = randCent(X,K)
    centroids,datalabel = kmeans(X,K)
    show(X, K, centroids, datalabel)
