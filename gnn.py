import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        print(edge_index)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


input_dim = 29  # 输入维度
hidden_dim = 64  # 隐藏维度
output_dim = 2  # 输出维度
gnn_model = GNNModel(input_dim, hidden_dim, output_dim)


def train(train_data):
    num_epochs = 100
    learning_rate = 0.01
    optimizer = optim.Adam(gnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    gnn_model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = gnn_model(train_data.x, train_data.edge_index)
        loss = criterion(output, train_data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


def modelEval(test_data):
    gnn_model.eval()

    with torch.no_grad():
        output = gnn_model(test_data.x, test_data.edge_index)
        _, predicted = torch.max(output, dim=1)
        accuracy = (predicted == test_data.y).sum().item() / len(test_data.y)
        print(f"Validation Accuracy: {accuracy}")
