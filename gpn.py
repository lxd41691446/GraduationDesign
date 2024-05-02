import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.utils import to_dense_batch


# 定义Graph Pooling Networks模型
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
