# Ali Jaabous
# Graph Sage Network
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html

from torch_geometric.nn import SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_predictor import Predictor

class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels*1, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels*1, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, train_data, hidden_channels_encoder, latent_space_dim, 
                 hidden_channels_predictor, n_classes, dropout_rate=0.5):
        super().__init__()
        self.encoder = Encoder(hidden_channels_encoder, latent_space_dim, dropout_rate)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.predictor = Predictor(latent_space_dim, hidden_channels_predictor, n_classes, dropout_rate)

    def forward(self, x, edge_index, edge_label_index, edge_weight):
        x = self.encoder(x, edge_index, edge_weight)
        x = self.predictor(x, edge_label_index)
        return x
