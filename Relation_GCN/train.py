import numpy as np
from dgl.contrib.data import load_data
import torch
from dgl import DGLGraph
from Relation_GCN.model import Model
graph=load_data(dataset='aifb')
num_nodes=graph.num_nodes
num_rels=graph.num_rels
labels=graph.labels
device='cpu'
edge_type=graph.edge_type
edge_norm=graph.edge_norm
# graph=graph.to(device)
g = DGLGraph((graph.edge_src, graph.edge_dst))
g.edata.update({'rel_type': torch.from_numpy(edge_type), 'norm': torch.from_numpy(edge_norm).unsqueeze(1)})
model=Model(1,num_rels,len(g))
model=model.to(device)
optim=torch.optim.Adam(model.parameters(),lr=1e-2)
train_index=torch.tensor([i for i in range(num_nodes//5*4)])
test_index=torch.tensor([i for i in range(num_nodes//5*4,num_nodes)])
loss_function=torch.nn.CrossEntropyLoss()
labels = torch.from_numpy(labels).view(-1)
for i in range(50):
    model(g)
    predict=g.ndata['h']
    loss=loss_function(predict[train_index],labels[train_index])
    accuracy=(torch.argmax(predict[train_index],-1)==labels[train_index]).sum()/train_index.shape[0]
    print("train {} epoch accuracy:{} loss{}".format(i,accuracy,loss.item()))
    loss.backward()
    optim.step()
    optim.zero_grad()
    with torch.no_grad():
        loss=loss_function(predict[test_index],labels[test_index])
        accuracy = torch.tensor(torch.argmax(predict[test_index], -1) == labels[test_index]).sum() / test_index.shape[0]
        print("eval {} epoch accuracy:{} loss{}".format(i, accuracy, loss.item()))




