import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='/home/jcosta/data/Planetoid', name='PubMed')

batch_size = 16  # Ajuste o tamanho do batch aqui
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Graph: {dataset[0]}')

data = dataset[0]

print(f'x = {data.x.shape}')
print(data.x)

print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)

data = data.to(device)

print(f'y = {data.y.shape}')
print(data.y)

print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)

print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')




class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(500, 256)
        self.gcn2 = GCNConv(256, 256)
        self.gcn3 = GCNConv(256, 128)
        self.gcn4 = GCNConv(128, 128)
        self.gcn5 = GCNConv(128, 64)
        self.gcn6 = GCNConv(64, 64)
        self.gcn7 = GCNConv(64, 32)
        self.out = torch.nn.Linear(32, 3)
        self.dropout = torch.nn.Dropout(p=0.5) 

    def forward(self, x, edge_index):
       
        h1 = F.relu(self.gcn1(x, edge_index))
        h1 = self.dropout(h1)

        
        h2 = F.relu(self.gcn2(h1, edge_index) + h1)
        h2 = self.dropout(h2)

        
        h3 = F.relu(self.gcn3(h2, edge_index))
        h3 = self.dropout(h3)

        
        h4 = F.relu(self.gcn4(h3, edge_index) + h3)
        h4 = self.dropout(h4)

        
        h5 = F.relu(self.gcn5(h4, edge_index))
        h5 = self.dropout(h5)

       
        h6 = F.relu(self.gcn6(h5, edge_index) + h5)
        h6 = self.dropout(h6)

        
        h7 = F.relu(self.gcn7(h6, edge_index))
        h7 = self.dropout(h7)

        z = self.out(h7)
        
        return h7, z

model = GCN().to(device)

print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

embeddings = []
losses = []
accuracies = []
outputs = []

for epoch in range(201):
    for data in loader:
       
        data = data.to(device)

     
        optimizer.zero_grad()

       
        h, z = model(data.x, data.edge_index)

        
        loss = criterion(z, data.y)

        acc = accuracy(z.argmax(dim=1), data.y)

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')