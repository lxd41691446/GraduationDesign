import torch
from torch_geometric.nn import GCNConv


# 初始化 GCN 模型
class FedGCN:
    # 随机初始化全局参数
    def random_initialize_global_params(model):
        for param in model.parameters():
            param.data = torch.randn_like(param.data)



