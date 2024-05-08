import torch
import torch.nn as nn
import torch.optim as optim
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
output_dim = 2  # 输出维度
# 用户模型初始化
gnn_model = GNNModel(input_dim, hidden_dim, output_dim)
# 全局模型初始化
global_model = GNNModel(input_dim, hidden_dim, output_dim)
# 调用函数进行全局参数随机初始化
fedAvg.FedGCN.random_initialize_global_params(global_model)

# 联邦学习迭代轮数
num_round = 10

def fed_train(train_list):
    num_epochs = 40
    learning_rate = 0.01
    # 定义优化器
    optimizer = optim.Adam(gnn_model.parameters(), lr=learning_rate)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 各用户的参数列表
    user_dict_list = [None] * dataSet.num_user

    gnn_model.train()

    for round in range(num_round):  # 联邦参数更新轮
        for index, item in enumerate(train_list):  # 遍历各个用户图
            # 将全局模型参数传递给用户模型
            gnn_model.load_state_dict(global_model.state_dict())
            for epoch in range(num_epochs):  # 每个用户模型进行训练
                optimizer.zero_grad()
                output = gnn_model(item.x, item.edge_index)
                loss = criterion(output, item.y)
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print(f"Epoch: {epoch + 1},round:{round+1} Loss: {loss.item()}")

            updated_dict = gnn_model.state_dict()
            # 写入更新的本地模型参数列表
            user_dict_list[index] = updated_dict
        #  执行参数更新
        aggregated_params = {}
        for params_dict in user_dict_list:
            # 在每个字典中遍历参数名称和对应的张量值
            for param_name, param_tensor in params_dict.items():
                # 如果参数名称在 aggregated_params 中不存在，则将其添加到 aggregated_params 中
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = param_tensor.clone().detach()
                # 否则，将参数张量值累加到 aggregated_params 中对应的张量上
                else:
                    aggregated_params[param_name] += param_tensor
        # 更新参数到全局模型上
        for param_name, param_tensor in aggregated_params.items():
            global_model_param = global_model.state_dict()[param_name]
            global_model_param.copy_(param_tensor)
    # 输出最后的全局模型量
    # print(global_model.state_dict())


def train(train_data):
    num_epochs = 100
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

    with torch.no_grad():
        output = gnn_model(test_data.x, test_data.edge_index)
        _, predicted = torch.max(output, dim=1)
        accuracy = (predicted == test_data.y).sum().item() / len(test_data.y)
        print(f"Validation Accuracy: {accuracy}")
