'''
实现logRegressClassifer，用于分类，损失函数使用交叉熵
参数优化，使用三种方法，普通梯度下降(BGD.适用于小数量)，
随机梯度下降(SGD,使用与大规模数据量)，批量梯度下降(MBGD,适用于通常规模数据量)
'''
import numpy as np
import matplotlib.pyplot as plt
import random
import time
##计算sigmode函数
def sigmode(x):
    return 1/(1+np.exp(-x))

def log_likelihood(x,y,weights):
    '''求解交叉熵损失的似然函数（也是损失函数），目标是最大化似然函数'''
    score = np.dot(x,weights)
    ##参考李航《统计机器学习》p79
    ll = np.sum(y*score)-np.log(1+np.exp(score))
    return ll

def cacal_gradient(x,y,y_pre):
    ##通过求解ll对weight的一阶偏导
    return np.dot(y-y_pre,x)

def LogRegrssClassifer(train_x,train_y,maxIter,learning_rate,optimizeType):
    numSample,numFeature = train_x.shape[0],train_x.shape[1]
    alpha =  learning_rate ##学习率
    maxIter = maxIter
    optimizeType = optimizeType
    weights = np.zeros(numFeature)  ##权重

    if optimizeType == "BGD":  ##普通梯度下降，每次使用所用样本对参数进行更新
        for inters in range(maxIter):
            score = np.dot(train_x, weights)  ##dot函数有顺序之分
            y_pre = sigmode(score)
            ##使用梯度下降更新模型
            gradient = cacal_gradient(train_x,train_y,y_pre)
            ##更新weight
            weights = weights+alpha*gradient  ##因为是求解似然函数最大化，所以是梯度上升，如果损失函数加上负号，就可以转化为梯度下降
            if inters%1000==0:
                print(log_likelihood(train_x,train_y,weights))
    elif optimizeType=="SBGD":
        for inters in range(maxIter):
            score = np.dot(train_x, weights)  ##dot函数有顺序之分
            y_pre = sigmode(score)
            index = random.randint(0,numSample-1)
            gradient = cacal_gradient(train_x[index,:],train_y[index,],y_pre[index,])
            weights = weights+alpha*gradient
    elif optimizeType=="Mini-BGD":##随机取batch个样本，而不是1个样本即可
        permutation = list(np.random.permutation(numSample))
        shuffled_X = train_x[permutation,:]
        shuffled_Y = train_y[permutation,:]
        mini_batch_size = 200  ##batch大小根据实际取值来定
        num_complete_minibatches = numSample / mini_batch_size  ##每个batch的样本个数
        mini_batches = []
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if numSample % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)


        score = np.dot(train_x, weights)
        y_pre = sigmode(score)
        index = random.randint(0, numSample)
        gradient = cacal_gradient(train_x[index, :], train_y[index, :], y_pre[index, :])
        weights = weights + alpha * gradient
    return weights

def predict(x,weights):
    y_pre = np.dot(x,weights)
    return y_pre

def log_loss(predicted, target):
    '''使用log评估模型的分类效果
    predicted是预测得到的结果
    target是真实的结果
    '''
    if len(predicted) != len(target):
        print('lengths not equal!')
        return
    target = [float(x) for x in target]  # make sure all float values,
    predicted = [min([max([x, 1e-15]), 1 - 1e-15]) for x in predicted]  # within (0,1) interval
    return -(1.0 / len(target)) * sum([target[i] * np.log(predicted[i]) + \
                                       (1.0 - target[i]) * np.log(1.0 - predicted[i]) \
                                       for i in range(len(target))])


def LogRegression(train_x,train_y,maxIter,learning_rate,optimizeType):
    numSample, numFeature = train_x.shape[0], train_x.shape[1]
    alpha = learning_rate  ##学习率
    maxIter = maxIter
    optimizeType = optimizeType
    weights = np.zeros(numFeature)  ##权重

    for inters in range(maxIter):
        y_pre = sigmode(np.dot(train_x, weights))
        error = train_y - y_pre
        if optimizeType == "BGD":  ##普通梯度下降，每次使用所用样本对参数进行更
            ##更新weight
            weights = weights + alpha *train_x* error  ##因为是求解似然函数最大化，所以是梯度上升，如果损失函数加上负号，就可以转化为梯度下降
            if inters % 1000 == 0:
                print(log_likelihood(train_x, train_y, weights))

    return weights


if __name__=="__main__":
    ###模拟输入数据
    np.random.seed(12)
    num_observations = 5000
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)
    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))
    plt.figure(figsize=(12, 8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=simulated_labels, alpha=.4)
    plt.show()
    intercept = np.ones((simulated_separableish_features.shape[0], 1)) ##设置偏置项
    features = np.hstack((intercept, simulated_separableish_features)) ##将偏置项与特征合并
    weights = LogRegrssClassifer(features, simulated_labels, maxIter=300000, learning_rate=5e-5,optimizeType='SBGD' )
    y_pre = predict(features,weights)
    logloss = log_loss(y_pre,simulated_labels)

    print(logloss)
