import numpy as np

def loadDataSet():
    import jieba
    postlist = [' 数据挖掘中的文本挖掘不论是对于企业应用，还是研究者工作，或者是参与数据竞赛项目，都是基础的工作。通过前面的一些实践工作，现总结出文本挖掘文本处理的通用流程。','注意，这里的文本挖掘任务主要指的是如文本分类、文本聚类、信息抽取、情感分类等等的常规NLP问题。','一、获取语料获取文本语料通常有以下几种方式：','1. 标准开放公开测试数据集，比如国内的中文汉语有搜狗语料、人民日报语料；国际English的有stanford的语料数据集、semavel的数据集等等','2. 爬虫抓取，获取网络文本，主要是获取网页HTML的形式，利用网络爬虫在相关站点爬取目标文本数据。']
    label = [1,0,1,0,0]
    cut_list = []
    for post_ in postlist:
        res = jieba.cut(post_)
        cut_list.append([res.strip() for res in list(res) if res.strip()])
    return cut_list,label

def createVocablist(corpus):
    res = set()
    for one_courpus in corpus:
        for word in one_courpus:
            res.add(word)
    return list(res)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''
    p（c_i|w） = p(w|c_i)p(c_i)/p(w)  ##分母在所有情况下都相等，所以不考虑分母

    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    numDoc, num_words = train_matrix.shape[0],train_matrix.shape[1]
    pAbusive = sum(trainCategory)/numDoc  ##p(c=1)
    p0Num = np.ones(num_words)
    p1Num = np.ones(num_words)
    laplace = 2.0 ##平滑项
    p0Denom = 0.0+laplace
    p1Denom = 0.0+laplace
    for i in range(numDoc):
        if trainCategory[i]==1:##是垃圾文本
            p1Num += trainMatrix[i]  ##没个词属于垃圾文本的次数
            p1Denom += sum(trainMatrix[i])  ##总的垃圾文字个数
        else:
            p0Num += trainMatrix[i] ##每个词不属于垃圾文本的次数
            p0Denom += sum(trainMatrix[i])  ##总的不是垃圾文字的个数
    p1Vect = np.log(p1Num/p1Denom)  ##p(x=x_i|c=1)
    p0Vect = np.log(p0Num/p0Denom ) ##p(x=x_i|c=2)  取对数，避免相乘下溢
    return p1Vect,p0Vect,pAbusive

def classifyNB(corpusMatrix,p0Vect,p1Vect,pClass1):
    p1 = sum(corpusMatrix*p1Vect)+np.log(pClass1)
    print(p1)
    p0 = sum(corpusMatrix*p0Vect+np.log(1.0-pClass1))
    if p1>p0:
        return 1
    else:
        return 0



if __name__=="__main__":
    cut_list,label = loadDataSet()
    vocab = createVocablist(cut_list)
    train_matrix = np.zeros((len(cut_list),len(vocab)))
    for index,cut_word in enumerate(cut_list):
        train_matrix[index,:] = np.array(setOfWords2Vec(vocab,cut_word))
    p1Vect,p0Vect,pAbusive = trainNB0(train_matrix,label)

    newDoc = ['文本挖掘','不是','企业','的']
    newDocMatrix = np.array(setOfWords2Vec(vocab,newDoc))
    print(classifyNB(newDocMatrix,p0Vect,p1Vect,pAbusive))