import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# 定义 GAT 模型
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj_matrix):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [GATLayer(in_features, hidden_features, dropout=dropout, alpha=alpha) for _ in range(num_heads)])
        self.out_att = GATLayer(hidden_features * num_heads, out_features, dropout=dropout, alpha=alpha)

    def forward(self, input, adj_matrix):
        x = F.dropout(input, self.dropout, training=self.training)
        x = torch.cat([att(x, adj_matrix) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj_matrix)
        return F.log_softmax(x, dim=1)


# 初始化模型和优化器
in_features = 29
hidden_features = 8
out_features = 2
num_heads = 4
dropout = 0.6
alpha = 0.2

gat_model = GAT(in_features, hidden_features, out_features, num_heads, dropout, alpha)
optimizer = optim.Adam(gat_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(train_loader, test_loader, adj_train_matrix, adj_test_matrix):
    # 训练模型
    num_epochs = 10
    # 将稀疏矩阵展开
    num_train_nodes = adj_train_matrix.shape[0]
    num_train_edges = adj_train_matrix.nnz
    adj_train_sparse = adj_train_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((adj_train_sparse.row, adj_train_sparse.col))).long()
    values = torch.from_numpy(adj_train_sparse.data).float()
    adj = torch.sparse_coo_tensor(indices, values, torch.Size(adj_train_sparse.shape))

    num_test_nodes = adj_test_matrix.shape[0]
    num_test_edges = adj_test_matrix.nnz
    adj_test_sparse = adj_test_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((adj_test_sparse.row, adj_test_sparse.col))).long()
    values = torch.from_numpy(adj_test_sparse.data).float()
    adj = torch.sparse_coo_tensor(indices, values, torch.Size(adj_test_sparse.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat_model.to(device)

    for epoch in range(num_epochs):
        gat_model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = gat_model(batch_inputs, adj_train_matrix)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_labels).sum().item()
            train_acc += correct / batch_labels.size(0)

            train_loss += loss.item()

            # 在测试集上评估模型
        gat_model.eval()
        test_loss = 0.0
        test_acc = 0.0

        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                outputs = gat_model(batch_inputs, adj_test_matrix)
                loss = criterion(outputs, batch_labels)

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == batch_labels).sum().item()
                test_acc += correct / batch_labels.size(0)

                test_loss += loss.item()

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        # 打印训练过程中的指标
        print(f"Epoch: {epoch + 1}/{num_epochs},"
              f" Train Loss: {train_loss:.4f},"
              f" Train Acc: {train_acc:.4f},"
              f" Test Loss: {test_loss:.4f},"
              f" Test Acc: {test_acc:.4f}")
