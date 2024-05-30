import dataSet
import dataProcess
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import dataToGraph
import gat
import gcn
import gpn
import graphSAGE
import GANF

if __name__ == '__main__':
    csv_path = "dataProcess/resampled_data.csv"
    save_dir = 'Part_Data'
    user_graph_list = []
    user_gat_adjMatrix_list = []

    batch_size = 128  # 小批量大小

    dataProcess.get_label_data("dataProcess/resampled_data.csv")  # 数据打乱
    dataProcess.Tc_Part().split_csv(dataSet.File_Upset, train_set=dataSet.Train_Set, file_name=dataSet.File_Upset)  # 数据切分
    dataProcess.PyCSV().split_csv(csv_path=csv_path, save_dir=save_dir, num_user=dataSet.num_user)  # 模拟分配给各用户机器
    df = pd.read_csv(dataSet.File_Check)  # 转换测试集dataframe
    print(len(df))
    # GCN测试集部分
    testGraph = dataToGraph.Graph.turn_Graph(df)  # 测试集图生成
    # GCN训练部分
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/resampled_data_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        user_graph_list.append(trainGraph)
        # gcn.modelEval(testGraph, gcn.train(trainGraph))
    print("Now is GCN training")
    gcn.fed_train(user_graph_list, testGraph)

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
    print("Now is GraphSAGE training")
    # SAGE训练部分(单机）
    '''
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainSageGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        user_sage_graph_list.append(trainSageGraph)
        # graphSAGE.train(trainSageGraph, testSageGraph)
    '''
    # SAGE联邦学习
    graphSAGE.fed_train(user_graph_list, testGraph)
    print("Now is GPN training")
    '''
    # gpn训练部分（单机）
    testGpn = dataToGraph.Graph.turn_Graph(df)
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGpn = dataToGraph.Graph.turn_Graph(dfTrain)
        user_graph_list.append(trainGpn)
    # gpn.train_test(user_gpn_graph_list, testGpn)
    '''
    # gpn联邦学习
    gpn.fed_train(user_graph_list, testGraph)
    print("Now is GANF training")
    # 核心算法GANF训练
    '''
    testGANF = dataToGraph.Graph.turn_Graph(df)
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGANF = dataToGraph.Graph.turn_Graph(dfTrain)
        user_graph_list.append(trainGANF)
    # GANF.train_test(user_ganf_graph_list, testGANF)
    '''
    # GANF联邦学习
    GANF.fed_train(user_graph_list, testGraph)

    # 第二数据集训练
    print("now is the second dataset train")
    user_graph_second_list = []
    dataProcess.get_label_data("Data/creditcard_2023.csv")  # 数据打乱
    dataProcess.Tc_Part().split_csv(dataSet.File_Upset, train_set=dataSet.Train_Set_s, file_name=dataSet.File_Upset)  # 数据切分
    dataProcess.PyCSV().split_csv(csv_path="Data/creditcard_2023.csv", save_dir=save_dir, num_user=dataSet.num_user)  # 模拟分配给各用户机器
    df = pd.read_csv(dataSet.File_Check)  # 转换测试集dataframe
    print(len(df))
    # GCN测试集部分
    testGraphSecond = dataToGraph.Graph.turn_Graph(df)  # 测试集图生成
    # GCN训练部分
    for i in range(dataSet.num_user_s):
        dfTrain = pd.read_csv('Part_Data/creditcard_2023_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        user_graph_second_list.append(trainGraph)
    print("Now is GCN training")
    # 加入num参数以确认正在训练第二个数据集
    gcn.fed_train(user_graph_second_list, testGraphSecond, num=2)
    print("Now is SAGE training")
    graphSAGE.fed_train(user_graph_second_list, testGraphSecond, num=2)
    print("Now is GPN training")
    gpn.fed_train(user_graph_second_list, testGraphSecond, num=2)
    print("Now is GANF training")
    GANF.fed_train(user_graph_second_list, testGraphSecond, num=2)
