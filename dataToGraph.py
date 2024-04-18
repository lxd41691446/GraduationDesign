import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def turn_Graph(df):
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
        for i in range(k):
            cluster_indices = np.where(cluster_labels == i)[0]
            re_node = cluster_indices[0]
            for j in range(1, len(cluster_indices)):
                edge_index.append([re_node, cluster_indices[j]])
        edge_index = torch.tensor(np.array(edge_index).T)
        data = Data(x=x,y=y,edge_index=edge_index)
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

if __name__ == '__main__':
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'label': [0, 1, 0]
    })
    Graph.turn_Graph(df)
