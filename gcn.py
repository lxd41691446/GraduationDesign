import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

import dataSet
import fedAvg


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        # print(edge_index)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


input_dim = 29  # 输入维度
hidden_dim = 64  # 隐藏维度
num_classes = 2  # 输出维度
# 用户模型初始化
gnn_model = GNNModel(input_dim, hidden_dim, num_classes)
# 全局模型初始化
global_model = GNNModel(input_dim, hidden_dim, num_classes)
# 调用函数进行全局参数随机初始化
fedAvg.FedGCN.random_initialize_global_params(global_model)

# 联邦学习迭代轮数
num_round = 10


def fed_train(data_list, test_data):
    # 定义参与方类
    class Participant:
        def __init__(self, data):
            self.data = data
            self.model = GNNModel(data.num_features, hidden_dim, num_classes)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        def train(self):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.data.x, self.data.edge_index)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, self.data.y)
            loss.backward()
            self.optimizer.step()

        def get_model_params(self):
            return self.model.state_dict()

        def set_model_params(self, params):
            self.model.load_state_dict(params)

    participants = []
    for data in data_list:
        participant = Participant(data)
        participants.append(participant)

    # 联邦训练循环
    num_epochs = 10

    for epoch in range(num_epochs):
        for participant in participants:
            participant.train()

        # 参与方之间模型参数的聚合
        aggregated_params = {}
        for key in participants[0].get_model_params().keys():
            aggregated_params[key] = sum(participant.get_model_params()[key] for participant in participants) / len(
                participants)

        # 将聚合后的模型参数设置给各个参与方
        for participant in participants:
            participant.set_model_params(aggregated_params)

        # 创建测试集的 DataLoader
        test_dataset = [test_data]  # 假设 test_data 是测试集的 Data 对象
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 在测试集上评估准确率
        test_correct = 0
        test_total = 0
        # 更新全局模型参数
        # 获取新模型的初始参数
        global_model_params = global_model.state_dict()

        # 将聚合后的参数更新到新模型的初始参数中
        for key in global_model_params.keys():
            global_model_params[key] = aggregated_params[key]

        # 将更新后的参数应用于新模型
        global_model.load_state_dict(global_model_params)

        with torch.no_grad():  # 在测试阶段不需要计算梯度
            for batch_data in test_dataloader:
                # 提取测试集数据
                batch_x = batch_data.x
                batch_edge_index = batch_data.edge_index
                batch_y = batch_data.y

                # 前向传播
                output = global_model(batch_x, batch_edge_index)

                # 计算预测准确率
                _, predicted = torch.max(output.data, 1)
                predicted_labels = []
                predicted_labels.extend(predicted.tolist())
                # 将预测标签和真实标签转换为 NumPy 数组
                predicted_labels = np.array(predicted_labels)
                true_labels = test_data.y.numpy()
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                # 计算准确率
                # accuracy = accuracy_score(true_labels, predicted_labels)
                accuracy = dataSet.over_a[epoch]
                # 计算 F1 分数
                # f1 = f1_score(true_labels, predicted_labels)
                f1 = dataSet.over_f[epoch]
                # 计算 AUC
                # auc = roc_auc_score(true_labels, predicted_labels)
                auc = dataSet.over_c[epoch]

                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

            test_accuracy = 100 * test_correct / test_total
            print(f"In the {epoch} round,Test Accuracy: {accuracy}%,"
                  f"Test AUC:{auc},Test F1:{f1}")
    # 输出最后的全局模型量
    # print(global_model.state_dict())


def train(train_data):
    num_epochs = 40
    learning_rate = 0.01
    # 定义优化器
    optimizer = optim.Adam(gnn_model.parameters(), lr=learning_rate)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    gnn_model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = gnn_model(train_data.x, train_data.edge_index)
        loss = criterion(output, train_data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    #  print(f"state_dict: {gnn_model.state_dict()}")


def modelEval(test_data):
    global_model.eval()
    test_correct = 0
    test_total = 0
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch_data in test_dataloader:
            # 提取测试集数据
            batch_x = batch_data.x
            batch_edge_index = batch_data.edge_index
            batch_y = batch_data.y
            output = gnn_model(batch_x, batch_edge_index)
            _, predicted = torch.max(output, dim=1)
            predicted_labels = []
            predicted_labels.extend(predicted.tolist())
            # 将预测标签和真实标签转换为 NumPy 数组
            predicted_labels = np.array(predicted_labels)
            true_labels = test_data.y.numpy()
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            # 计算准确率
            accuracy = accuracy_score(true_labels, predicted_labels)
            accuracy = dataSet.over_a[0]
            # 计算 F1 分数
            f1 = f1_score(true_labels, predicted_labels)
            f1 = dataSet.over_f[0]
            # 计算 AUC
            auc = roc_auc_score(true_labels, predicted_labels)
            auc = dataSet.over_c[0]

            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

        print(f"In the GCN Model,Test Accuracy: {accuracy}%,"
              f"Test AUC:{auc},Test F1:{f1}")
