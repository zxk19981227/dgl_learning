from dgl.contrib.data import load_data
import torch
from Relation_GCN.model import Model
graph=load_data(dataset='aifb')
num_nodes=graph.num_nodes
num_rels=graph.num_rels
labels=graph.labels
device='cpu'

model=Model(1,num_rels)
model=model.to(device)
optim=torch.optim.Adam(model.parameters(),lr=1e-3)
train_index=torch.tensor([i for i in range(num_nodes//5*4)])
test_index=torch.tensor([i for i in range(num_nodes//5*4,num_nodes)])

for i in range(50):
    predcit=model()




