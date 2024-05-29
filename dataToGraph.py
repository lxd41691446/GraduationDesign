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
        n_samples, n_features = X.shape

        # 随机初始化 k 个簇中心
        centroids = X[torch.randperm(n_samples)[:n_clusters]]

        for _ in range(max_iter):
            # 为每个样本分配最近的簇标签
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)

            # 更新每个簇的质心
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(n_clusters)])

            # 判断是否达到收敛条件
            if torch.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        '''
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
            '''
        return labels

    def create_edge_index_from_clusters(labels):
        """
        根据聚类标签,创建 PyTorch Geometric 格式的 edge_index 张量。

        参数:
        labels (torch.Tensor): 每个样本所属的簇标签,shape为(n_samples,)。

        返回:
        edge_index (torch.LongTensor): 表示边的起始节点和终止节点的索引张量,shape为(2, n_edges)。
        """
        n_samples = labels.shape[0]

        # 创建一个空的 edge_index 张量
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        # 遍历所有样本,创建同一簇内的边
        for cluster_id in torch.unique(labels):
            # 找到属于当前簇的样本索引
            cluster_indices = torch.nonzero(labels == cluster_id).squeeze()

            # 创建同一簇内所有样本之间的边
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    # 添加边的起始节点和终止节点索引
                    edge_index = torch.cat([edge_index, torch.tensor([[cluster_indices[i]], [cluster_indices[j]]])],
                                           dim=1)

        return edge_index

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
            # print(cluster_indices)
            if len(cluster_indices) != 0:
                re_node = cluster_indices[0]
                for j in range(1, len(cluster_indices)):
                    edge_index.append([re_node, cluster_indices[j]])
        edge_index = torch.tensor(np.array(edge_index).T)

        # edge_index = Graph.create_edge_index_from_clusters(cluster_labels)
        data = Data(x=x, y=y, edge_index=edge_index)
        print(data)
        print("done!")

        '''
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
