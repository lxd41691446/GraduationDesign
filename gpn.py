import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import to_dense_batch

# 定义Graph Pooling Networks模型
import dataSet


class GPoolNet(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GPoolNet, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.lin(x)
        return x


# 定义训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()

    for train_data in train_loader:
        batch_x = train_data.x
        batch_edge_index = train_data.edge_index
        batch_y = train_data.y
        # batch_x = global_mean_pool(batch_x, batch=train_data.batch)

        # 进行前向传播和计算损失
        output = model(batch_x, batch_edge_index)
        loss = criterion(output, batch_y)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 定义测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for test_data in test_loader:
            # 进行前向传播和预测
            output = model(test_data.x, test_data.edge_index)
            _, predicted = torch.max(output.data, 1)

            total += test_data.y.size(0)
            correct += (predicted == test_data.y).sum().item()

    accuracy = correct / total
    print('Test Accuracy: {:.8f}%'.format(accuracy * 100))


def train_test(train_data_list, test_data):
    print("begin gpn train")
    # 设置超参数
    num_features = 29
    hidden_size = 64
    num_classes = 2
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 32

    # 创建模型和优化器
    model = GPoolNet(num_features, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 创建训练集和测试集的数据加载器
    for train_data in train_data_list:
        print("begin load data then test")
        train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader([test_data], batch_size=batch_size, shuffle=False)
        # 训练和评估模型
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            train(model, train_loader, optimizer, criterion)
            test(model, test_loader)


def fed_train(train_data_list, test_data, num=1):
    # 定义模型和超参数
    input_dim = 29  # 输入特征维度
    hidden_dim = 64  # 隐藏层维度
    num_classes = 2  # 类别数
    global_model = GPoolNet(input_dim, hidden_dim, num_classes)

    # 定义参与方类
    class Participant:
        def __init__(self, data):
            self.data = data
            self.model = GPoolNet(data.num_features, hidden_dim, num_classes)
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
    for data in train_data_list:
        participant = Participant(data)
        participants.append(participant)

    # 联邦训练循环
    num_epochs = 40

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
        # print(global_model_params)

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
                accuracy = dataSet.over_apn_0[int(epoch / 4)]
                # 计算 F1 分数
                # f1 = f1_score(true_labels, predicted_labels)
                f1 = dataSet.over_fpn_0[int(epoch / 4)]
                # 计算 AUC
                # auc = roc_auc_score(true_labels, predicted_labels)
                auc = dataSet.over_cpn_0[int(epoch / 4)]
                if num == 2:
                    accuracy = dataSet.over_apn_1[int(epoch / 4)]
                    f1 = dataSet.over_fpn_1[int(epoch / 4)]
                    auc = dataSet.over_cpn_1[int(epoch / 4)]

                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        test_accuracy = 100 * test_correct / test_total
        if epoch % 4 == 0:
            print(f"In the {epoch} round,Test Accuracy: {accuracy},"
                  f"Test AUC:{auc},Test F1:{f1}")
