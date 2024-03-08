# Ali Jaabous
# Graph Attentional Network
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html

from torch_geometric.nn import GATConv, Linear, to_hetero, HeteroConv, GeneralConv, BatchNorm
from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, edge_dim=1, heads=3, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels*3, hidden_channels, edge_dim=1, heads=3, add_self_loops=False)
        self.conv3 = GATConv(hidden_channels*3, out_channels, edge_dim=1, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        x = F.relu(x)
        x= self.conv3(x, edge_index, edge_attr=edge_weight)
        return x

class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x['user'][row], x['movie'][col]], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, train_data, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes):
        super().__init__()
        self.encoder = Encoder(hidden_channels_encoder, latent_space_dim)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.predictor = Predictor(latent_space_dim, hidden_channels_predictor, n_classes)

    def forward(self, x, edge_index, edge_label_index, edge_weight):
        x = self.encoder(x, edge_index, edge_weight)
        x = self.predictor(x, edge_label_index)
        return x