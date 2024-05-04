import dataSet
import dataProcess
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import dataToGraph
import gat
import gcn
import gpn
import graphSAGE

if __name__ == '__main__':
    csv_path = dataSet.File_Train
    save_dir = 'Part_Data'
    user_graph_list = []
    user_gat_graph_list = []
    user_sage_graph_list = []
    user_gat_adjMatrix_list = []
    user_gpn_graph_list = []

    batch_size = 128  # 小批量大小

    dataProcess.get_label_data()  # 数据打乱
    dataProcess.Tc_Part().split_csv(dataSet.File_Upset)  # 数据切分
    dataProcess.PyCSV().split_csv(csv_path=csv_path, save_dir=save_dir)  # 模拟分配给各用户机器
    df = pd.read_csv(dataSet.File_Check)  # 转换测试集dataframe
    print(len(df))
    '''
    # GCN测试集部分
    testGraph = dataToGraph.Graph.turn_Graph(df)  # 测试集图生成
    # GCN训练部分
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        user_graph_list.append(trainGraph)
    gcn.fed_train(user_graph_list)
    gcn.modelEval(testGraph)
    '''
    '''
    # GAT测试集生成部分
    testGatGraph, testAdjMatrix = dataToGraph.Graph.turn_graph_gat(df)
    test_loader = DataLoader(testGatGraph, batch_size=batch_size, shuffle=False)
    # GAT训练部分
    for i in range(dataSet.num_user):
        dfTrainGat = pd.read_csv('Part_Data/Data_Train_' + str(i+1) + '.csv')
        trainGatGraph, trainAdjMatrix = dataToGraph.Graph.turn_graph_gat(dfTrainGat)
        train_loader = DataLoader(trainGatGraph, batch_size=batch_size, shuffle=True)
        user_gat_graph_list.append(trainGatGraph)
        user_gat_adjMatrix_list.append(trainAdjMatrix)
        gat.train(train_loader, test_loader, trainAdjMatrix, testAdjMatrix)
    '''
    '''
    # SAGE训练部分(单机）
    testSageGraph = dataToGraph.Graph.turn_Graph(df)  # 测试集图生成
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainSageGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        user_sage_graph_list.append(trainSageGraph)
        # graphSAGE.train(trainSageGraph, testSageGraph)
    # SAGE联邦学习
    graphSAGE.fed_train(user_sage_graph_list, testSageGraph)
    '''

    # gpn训练部分（单机）
    testGpn = dataToGraph.Graph.turn_Graph(df)
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGpn = dataToGraph.Graph.turn_Graph(dfTrain)
        user_gpn_graph_list.append(trainGpn)
    # gpn.train_test(user_gpn_graph_list, testGpn)
    # gpn联邦学习
    gpn.fed_train(user_gpn_graph_list, testGpn)



