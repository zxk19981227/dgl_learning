from GAT.GAT_model import GAT
import torch
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

data = citegrh.load_cora()
features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)
mask = torch.BoolTensor(data.train_mask)
g = DGLGraph(data.graph)
test_mask=torch.BoolTensor(data.test_mask)
model=GAT(g,features.shape[1],7,2)
optim=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_function=torch.nn.CrossEntropyLoss()
for i in range(1000):
    model.train()
    predict=model(features)
    loss=loss_function(predict[mask],labels[mask])
    loss.backward()
    optim.step()
    optim.zero_grad()
    train_predict=torch.argmax(predict[mask],-1)
    correct=(train_predict==labels[mask]).sum().item()
    total=(mask).sum()
    print("epoch {} train accuracy {} loss{}".format(i,correct/total,loss.item()))
    with torch.no_grad():
        loss = loss_function(predict[test_mask], labels[test_mask])
        train_predict = torch.argmax(predict[test_mask],-1)
        correct = (train_predict == labels[test_mask]).sum().item()
        total = (test_mask).sum()
        print("epoch {} eval accuracy {} loss{}".format(i, correct / total, loss.item()))