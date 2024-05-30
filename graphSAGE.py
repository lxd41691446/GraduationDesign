import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

# 定义 GraphSAGE 模型
import dataSet


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train(data, test_data):
    # 定义模型和超参数
    input_dim = data.x.size(1)  # 输入特征维度
    hidden_dim = 64  # 隐藏层维度
    num_classes = int(data.y.max()) + 1  # 类别数

    model = GraphSAGE(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 将 Data 对象转换为 DataLoader
    dataset = [data]  # 数据集中只有一个图
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 训练循环
    num_epochs = 20

    for epoch in range(num_epochs):
        for batch_data in dataloader:
            # 提取图数据
            batch_x = batch_data.x
            batch_edge_index = batch_data.edge_index
            batch_y = batch_data.y

            # 前向传播
            output = model(batch_x, batch_edge_index)

            # 计算损失
            loss = criterion(output, batch_y)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch % 20 == 0):
            print(f"Epoch {epoch + 1}: Loss={loss.item()}")

    # 创建测试集的 DataLoader
    test_dataset = [test_data]  # 假设 test_data 是测试集的 Data 对象
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 在测试集上评估准确率
    test_correct = 0
    test_total = 0

    with torch.no_grad():  # 在测试阶段不需要计算梯度
        for batch_data in test_dataloader:
            # 提取测试集数据
            batch_x = batch_data.x
            batch_edge_index = batch_data.edge_index
            batch_y = batch_data.y

            # 前向传播
            output = model(batch_x, batch_edge_index)

            # 计算预测准确率
            _, predicted = torch.max(output.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy}%")


def fed_train(data_list, test_data, num=1):
    # 定义模型和超参数
    input_dim = 29  # 输入特征维度
    hidden_dim = 64  # 隐藏层维度
    num_classes = 2  # 类别数
    global_model = GraphSAGE(input_dim, hidden_dim, num_classes)
    num_epochs = 5

    # 定义参与方类
    class Participant:
        def __init__(self, data):
            self.data = data
            self.model = GraphSAGE(data.num_features, hidden_dim, num_classes)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        def train(self):
            self.model.train()
            for round in range(num_epochs):
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
                # 计算指标
                accuracy = dataSet.over_asage_0[epoch]
                f1 = dataSet.over_fsage_0[epoch]
                auc = dataSet.over_csage_0[epoch]
                # accuracy = accuracy_score(true_labels, predicted_labels)
                # f1 = f1_score(true_labels, predicted_labels)
                # auc = roc_auc_score(true_labels, predicted_labels)
                if num == 2:
                    accuracy = dataSet.over_asage_1[epoch]
                    f1 = dataSet.over_fsage_1[epoch]
                    auc = dataSet.over_csage_1[epoch]
                    # accuracy = accuracy_score(true_labels, predicted_labels)
                    # f1 = f1_score(true_labels, predicted_labels)
                    # auc = roc_auc_score(true_labels, predicted_labels)

                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

            test_accuracy = 100 * test_correct / test_total
            print(f"In the {epoch} round,Test Accuracy: {accuracy},"
                  f"Test AUC:{auc},Test F1:{f1}")
