import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader


# 定义 GraphSAGE 模型
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
    num_epochs = 40

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
        if(epoch % 20 == 0):
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
