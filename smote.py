# 实现负样本的扩容smote算法
import dataProcess
import dataSet
import csv
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from numpy import genfromtxt
import numpy as np



class Smote(object):
    """
    SMOTE algorithm implementation.
    Parameters
    ----------
    samples : {array-like}, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.
    N : int, optional (default = 50)
        Parameter N, the percentage of n_samples, affects the amount of final
        synthetic samples，which calculated by floor(N/100)*T.
    k : int, optional (default = 5)
        Specify the number for NearestNeighbors algorithms.
    r : int, optional (default = 2)
        Parameter for sklearn.neighbors.NearestNeighbors API.When r = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for r = 2. For arbitrary p, minkowski_distance (l_r) is used.
    Examples
    --------
      >>> samples = np.array([[3,1,2], [4,3,3], [1,3,4],
                              [3,3,2], [2,2,1], [1,4,3]])
      >>> smote = Smote(N=200)
      >>> synthetic_points = smote.fit(samples)
      >>> print(synthetic_points)
      [[3.31266454 1.62532908 2.31266454]
       [2.4178394  1.5821606  2.5821606 ]
       [3.354422   2.677211   2.354422  ]
       [2.4169074  2.2084537  1.4169074 ]
       [1.86018171 2.13981829 3.13981829]
       [3.68440949 3.         3.10519684]
       [2.22247957 3.         2.77752043]
       [2.3339721  2.3339721  1.3339721 ]
       [3.31504371 2.65752185 2.31504371]
       [2.54247589 2.54247589 1.54247589]
       [1.33577795 3.83211103 2.83211103]
       [3.85206355 3.04931215 3.        ]]
    """

    def __init__(self, N=50, k=5, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        # self.r是距离决定因子
        self.r = r
        # self.newindex用于记录SMOTE算法已合成的样本个数
        self.newindex = 0

    # 构建训练函数
    def fit(self, samples):
        # 初始化self.samples, self.T, self.numattrs
        self.samples = samples
        # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
        self.T, self.numattrs = self.samples.shape

        # 查看N%是否小于100%
        if self.N < 100:
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            # N%变成100%
            self.N = 100

        # 查看从T是否不大于近邻数k
        if self.T <= self.k:
            # 若是，k更新为T-1
            self.k = self.T - 1

        # 令N是100的倍数
        N = int(self.N / 100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)

        # 对所有输入样本做循环
        for i in range(len(self.samples)):
            # 调用kneighbors方法搜索k近邻
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]

            # 把N,i,nnarray输入样本合成函数self.__populate
            self.__populate(N, i, nnarray)

        # 最后返回合成样本self.synthetic
        return self.synthetic

    # 构建合成样本函数
    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k - 1)

            # 计算差值
            diff = self.samples[nnarray[nn]] - self.samples[i]
            # 随机生成一个0～1之间的数
            gap = random.uniform(0, 1)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = self.samples[i] + gap * diff
            # print(type(self.samples[i]+gap * diff))
            # print(self.samples[i])
            # print(self.synthetic[self.newindex])

            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1


def less_make():
    # 将负样本完全提取出来
    rowuse = 0
    rowno = 0
    list = []
    # 首先将数据复制一份
    # with open(Data.File_Smote, 'ab') as f:
    #     f.write(open(Data.File_Name, 'rb').read())
    with open(dataSet.File_Name, encoding="utf-8", errors="ignore") as f:
        # 可通过列名读取列值，表中有空值
        data = csv.DictReader(_.replace("\x00", "") for _ in f)
        headers = next(data)
        print(headers)
        for row in data:
            rowno += 1
            # print(row)
            if row['Class'] == '1':
                rowuse += 1
                # print(row)
                list.append(rowno)
        print(rowuse)
        print(rowno)

    # 读取数据集
    data = pd.read_csv(dataSet.File_Name)
    for i in range(rowuse):
        use = int(list[i])
        output_file = data.loc[use:use]
        save = 'Data_Smote/smote' + str(i) + '.csv'
        output_file.to_csv(save, mode="w", encoding="utf-8", index=False)
    # 抽取出所有的负样本合并为Data_Less.csv
    dataProcess.PyCSV().merge_csv(save_name=dataSet.File_Less, file_dir="Data_Smote")


if __name__ == '__main__':
    less_make()
    samples = pd.read_csv("Data_Smote/Data_Less.csv")
    samples = np.array(samples)
    # print(samples)
    # N值决定负样本将会扩充至原样本的多少，如下N = 325代表将会扩充原来的325%负样本数量，但是实际上因为向下取整只会扩增至300%
    smote = Smote(N=1000)
    synthetic_points = smote.fit(samples)
    # print(synthetic_points)
    np.savetxt(dataSet.File_Smote, synthetic_points, delimiter=",")
    # 之后可以将该文件和原文件进行合并
    # 由于自带的merge函数无法支持合并一个浮点数据集和一个int数据集，所以我们使用批处理文件来进行工作
    # 我们先将原数据集复制到BorderLine_Smote路径下，生成的负样本集合也在该路径下
    # 批处理文件会将该路径下的所有数据集合并得到下面的data路径内的csv文件，在dataSet文件中该路径名为File_Merge_Smote
    # Data_Part.PyCSV().merge_csv(save_name='train.csv', file_dir="Train")
    data = pd.read_csv('dataProcess/smote.csv', sep='，')
    data.info()