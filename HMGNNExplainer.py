import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import k_hop_subgraph

from utils.dataset import HeterTUDataset
from utils.new_data import convert_data
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

loss_fn = torch.nn.CrossEntropyLoss()

data_name = 'PTC_MR'

if data_name == "Mutagenicity":
    num_nodes = 226
    start_index, end_index = num_nodes, num_nodes+3552
elif data_name == "PTC_MR":
    num_nodes = 95
    num_classes=2
    start_index, end_index = num_nodes, num_nodes+307
elif data_name == "NCI1":
    num_nodes = 150
    start_index, end_index = num_nodes, num_nodes
elif data_name == "NCI109":
    num_nodes = 150
    start_index, end_index = num_nodes, num_nodes

dataset = HeterTUDataset('data/' + data_name, data_name, num_nodes)
data = dataset[0]
# print(data.smiles_list)
data_smiles = data.smiles_list
# print(stop)

ori_dataset = TUDataset("data/", data_name)
dataloader = DataLoader(ori_dataset, batch_size=32)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 16
        self.conv1 = GCNConv(dataset.num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x



mask = torch.tensor([x for x in range(num_nodes,data.x.size(0))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
ori_dataset = ori_dataset
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = loss_fn(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

num_graphs = 0
fidelity = 0
for id in tqdm(range(len(data.selected_idx))):
    node_index = id+num_nodes
    out1 = model(data.x, data.edge_index)
    
    if torch.argmax(out1[node_index]) != data.y[node_index]:
        continue
    else:
        explanation = explainer(data.x, data.edge_index, index=node_index)

        selected_edges = []
        selected_values = []
        threshold = 0.0
        max_index = -1
        max_value = 0.0

        for i in range(explanation.edge_mask.size(0)):
            value = explanation.edge_mask[i].item()
            if value > threshold:
                selected_edges.append(i)
                selected_values.append(value)

        selected_atom = -1

        for i in range(len(selected_edges)):
            edge = selected_edges[i]
            edge_list = data.edge_index[:, edge].tolist()
            if node_index == edge_list[0]:
                if selected_values[i] > max_value:
                    selected_atom = edge_list[1]
            elif node_index == edge_list[1]:
                if selected_values[i] > max_value:
                    selected_atom = edge_list[0]
            else:
                continue

        edge_list = data.edge_index[:, edge].tolist()
        edge_mask = []
        for i in range(data.edge_index.size(1)):
            if (data.edge_index[0,i]==node_index and data.edge_index[1,i]!=selected_atom) or (data.edge_index[0,i]!=selected_atom and data.edge_index[1,i]==node_index):
                continue
            else:
                edge_mask.append(i)
        new_edge_index = data.edge_index[:,edge_mask]

        out2 = model(data.x, new_edge_index)
        num_graphs += 1

        fidelity += (out1[node_index].softmax(dim=0)[data.y[node_index]] - out2[node_index].softmax(dim=0)[data.y[node_index]])
    
fidelity = fidelity/num_graphs

print(f"Fidelity of dataset {data_name}: {fidelity}.")
