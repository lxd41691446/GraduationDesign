import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import networkx as nx


class Graph:
    def kmeans(X, n_clusters, max_iter=100):
        # 随机选择初始聚类中心
        random_indices = torch.randperm(X.shape[0])[:n_clusters]
        centroids = X[random_indices]

        for _ in range(max_iter):
            # 计算每个样本到聚类中心的距离
            distances = torch.norm(X[:, None] - centroids, dim=-1)

            # 分配样本到最近的聚类中心
            labels = torch.argmin(distances, dim=-1)

            # 更新聚类中心为每个簇的平均值
            new_centroids = torch.stack([torch.mean(X[labels == k], dim=0) for k in range(n_clusters)])

            # 判断是否收敛
            if torch.all(torch.eq(centroids, new_centroids)):
                break

            centroids = new_centroids

        return labels

    def turn_Graph(df):
        x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float)
        y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        k = int(len(df) / 1000)
        print("prepare make KMeans")
        '''
        kmeans = KMeans(n_clusters=k)
        data = Data(x=x, y=y)
        kmeans.fit(x)
        cluster_labels = kmeans.labels_
        '''
        cluster_labels = Graph.kmeans(x, k)
        edge_index = []
        print("prepare make edge")
        for i in range(k):
            cluster_indices = np.where(cluster_labels == i)[0]
            re_node = cluster_indices[0]
            for j in range(1, len(cluster_indices)):
                edge_index.append([re_node, cluster_indices[j]])
        edge_index = torch.tensor(np.array(edge_index).T)
        data = Data(x=x, y=y, edge_index=edge_index)
        print(data)
        print("done!")

        # 创建一个无向图对象
        graph = nx.Graph()

        # 添加节点
        for i in range(len(x)):
            graph.add_node(i, pos=(x[i], y[i]))

        # 添加边
        for i in range(edge_index.shape[1]):
            src = edge_index[0][i]
            tgt = edge_index[1][i]
            graph.add_edge(src, tgt)

        print("graph created")
        '''
        # 绘制图像
        pos = nx.spring_layout(graph)  # 选择一种布局算法
        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, with_labels=True, node_color=y, cmap='coolwarm', node_size=200)
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        sm.set_array(y)
        plt.colorbar(sm)
        # 显示图像
        plt.title("Graph Visualization")
        plt.savefig("a.png")
        # plt.show()
        '''
        return data

    def turn_graph_gat(df):
        # 生成聚类用于进行邻接矩阵的生成
        x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float)
        y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        k = int(len(df) / 1000)
        print("prepare make KMeans")
        kmeans = KMeans(n_clusters=k)
        data = Data(x=x, y=y)
        kmeans.fit(x)
        cluster_labels = kmeans.labels_
        edge_index = []
        print("prepare make edge")
        # 由于样本过大，采用生成稀疏矩阵的方法解决该问题
        for i in range(k):
            cluster_indices = np.where(cluster_labels == i)[0]
            re_node = cluster_indices[0]
            for j in range(1, len(cluster_indices)):
                edge_index.append([re_node, cluster_indices[j]])
        edge_index = torch.tensor(np.array(edge_index).T)

        num_nodes = len(x)
        num_edges = len(edge_index)

        # 构建 COO 格式的稀疏矩阵
        adj_matrix_sparse = coo_matrix((np.ones(num_edges), (edge_index[:, 0], edge_index[:, 1])),
                                       shape=(num_nodes, num_nodes))

        # 将稀疏矩阵转换为 CSR 格式
        adj_matrix_sparse = adj_matrix_sparse.tocsr()

        # 可以通过打印稀疏矩阵来查看其内容
        print(f"adj_matrix_sparse :{adj_matrix_sparse}")

        features = x
        labels = y
        dataset = TensorDataset(features, labels)
        return dataset, adj_matrix_sparse


if __name__ == '__main__':
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'label': [0, 1, 0]
    })
    Graph.turn_Graph(df)
