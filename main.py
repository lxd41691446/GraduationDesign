import dataSet
import dataProcess
import pandas as pd

import dataToGraph
import gnn

if __name__ == '__main__':
    csv_path = dataSet.File_Train
    save_dir = 'Part_Data'

    dataProcess.get_label_data()  # 数据打乱
    dataProcess.Tc_Part().split_csv(dataSet.File_Upset)  # 数据切分
    dataProcess.PyCSV().split_csv(csv_path=csv_path, save_dir=save_dir)  # 模拟分配给各用户机器
    df = pd.read_csv(dataSet.File_Check)  # 转换测试集dataframe
    print(len(df))
    testGraph = dataToGraph.Graph.turn_Graph(df)  # 测试集图生成
    for i in range(dataSet.num_user):
        dfTrain = pd.read_csv('Part_Data/Data_Train_' + str(i + 1) + '.csv')  # 生成各用户图
        trainGraph = dataToGraph.Graph.turn_Graph(dfTrain)
        gnn.train(trainGraph)
        gnn.modelEval(testGraph)

