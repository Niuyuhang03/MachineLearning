def create_C1(data):
    '''
    生成频繁项集C1
    :param data: 原始出数据
    :return: 频繁项集C1
    '''
    C1 = []
    for items in data:
        for item in items:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def aprioriGen(Lksub1, k):
    '''
    根据候选集L_(k-1)，生成频繁项集Ck
    :param Lk: 候选集L_(k-1)
    :param k: 编号k，即每个候选项的长度
    :return: 频繁项集Ck
    '''
    retList = []
    lenLk = len(Lksub1)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lksub1[i])[:k-2]
            L2 = list(Lksub1[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lksub1[i] | Lksub1[j])
    return retList

def scanD(D, Ck, minSupport):
    '''
    扫描数据集，得到下一个的频繁项集的候选集
    :param D:数据集
    :param Ck:当前频繁项集
    :param minSupport:最小支持度
    :return:新的频繁项集的候选集
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid): # 寻找子集
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def apriori(dataset, minSupport = 0.5):
    '''
    apriori算法，得到最后的频繁项集
    :param dataset: 数据集
    :param minSupport: 最小支持度
    :return: 最后一个频繁项集
    '''
    C1 = create_C1(dataset)
    D = list(map(set, dataset))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    '''
    生成关联规则
    :param L:频繁项集
    :param supportData:支持度集合
    :param minConf:最小置信度
    :return:关联规则列表
    '''
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    计算置信度，评估规则
    :param freqSet:一个频繁项
    :param H:一个频繁项的单个元素集合，其每一项都作为规则左部
    :param supportData:支持度集合
    :param brl:关联规则集合
    :param minConf:最小置信度
    :return:关联规则右部集合
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[conseq]
        if conf >= minConf:
            print(str(conseq)+'-->'+str(freqSet-conseq)+' conf:'+str(conf))
            brl.append(((freqSet-conseq, conseq, conf)))
            prunedH.append((conseq))
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    一个候选集内项目多于2时，生成候选规则集合
    :param freqSet:一个频繁项
    :param H:一个频繁项的单个元素集合，其每一项都作为规则左部
    :param supportData:支持度集合
    :param brl:关联规则集合
    :param minConf:最小置信度
    '''
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == '__main__':
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    L, supportData = apriori(dataset, 0.5)
    generateRules(L, supportData, 0.5)