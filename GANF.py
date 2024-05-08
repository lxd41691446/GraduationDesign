import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from NF import MAF, RealNVP
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch

import gcn


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class GANF(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, dropout=0.1, model="MAF", batch_norm=True):
        super(GANF, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')
        else:
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                              batch_norm=batch_norm)

    def forward(self, x, A):

        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D
        x = torch.unsqueeze(x, 2)  # 在第三个维度上增加一个维度
        x = torch.unsqueeze(x, 3)  # 在第四个维度上增加一个维度
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape([full_shape[0], -1])  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def locate(self, x, A):
        # x: N X K X L X D
        x = torch.unsqueeze(x, 2)  # 在第三个维度上增加一个维度
        x = torch.unsqueeze(x, 3)  # 在第四个维度上增加一个维度
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape([full_shape[0], full_shape[1], -1])  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=2)

        return log_prob


# 初始化 GANF 模型
n_blocks = 4
input_size = 29
hidden_size = 64
n_hidden = 2
dropout = 0.0
model = "MAF"
batch_norm = True

ganf = GANF(n_blocks, input_size, hidden_size, n_hidden, dropout, model, batch_norm)
gcnmodel = gcn.GNNModel(input_size, hidden_size, output_dim=2)


def train_test(train_list, test_data):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcnmodel.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcnmodel.to(device)

    # 创建训练集和测试集的数据加载器
    for train_data in train_list:
        print("begin load data then test")
        train_loader = DataLoader([train_data], batch_size=64, shuffle=True)
        # 训练和评估模型
        for epoch in range(num_epochs):
            gcnmodel.train()
            for data_train in train_loader:
                batch_x = data_train.x
                batch_y = data_train.y
                edge_index = data_train.edge_index
                # 前向传播
                A = torch.randn(input_size, input_size)
                output = gcnmodel(batch_x, edge_index)

                # 计算损失
                loss = criterion(output, batch_y)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}")
        test(test_data)


def test(test_data):
    criterion = nn.CrossEntropyLoss()
    # 在测试集上进行验证
    gcnmodel.eval()
    with torch.no_grad():
        output = gcnmodel(test_data.x, test_data.edge_index)
        _, predicted = torch.max(output, dim=1)
        accuracy = (predicted == test_data.y).sum().item() / len(test_data.y)
        loss = criterion(output, test_data.y)
        test_loss = loss.item()
        print(f"Validation Accuracy: {accuracy}")
        print(f"Test Loss: {test_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}")


def fed_train(train_data_list, test_data):
    # 定义模型和超参数
    input_dim = 29  # 输入特征维度
    hidden_dim = 64  # 隐藏层维度
    output_dim = 2  # 类别数
    global_model = gcn.GNNModel(input_dim, hidden_dim, output_dim)

    # 定义参与方类
    class Participant:
        def __init__(self, data):
            self.data = data
            self.model = gcn.GNNModel(data.num_features, hidden_dim, output_dim)
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
    num_epochs = 50

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
                # 计算准确率
                accuracy = accuracy_score(true_labels, predicted_labels)
                # 计算 F1 分数
                f1 = f1_score(true_labels, predicted_labels)
                # 计算 AUC
                auc = roc_auc_score(true_labels, predicted_labels)

                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        test_accuracy = 100 * test_correct / test_total
        print(f"In the {epoch} round,Test Accuracy: {accuracy}%,"
              f"Test AUC:{auc},Test F1:{f1}")
