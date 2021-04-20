from dgl.data import CoraGraphDataset
import torch
from graph_conv.gcn  import GCN
dataset=CoraGraphDataset()
g=dataset[0]#因为只有一个graph，所以直接写0
label=g.ndata['label']
features=g.ndata['feat']
train_mask=g.ndata['train_mask']
test_mask=g.ndata['test_mask']
def train(model:GCN,optim,loss_function,train_mask,g,label,features,step):
    predict=model(features,g)
    loss=loss_function(predict[train_mask],label[train_mask])
    loss.backward()
    optim.step()
    optim.zero_grad()
    accuracy=(predict[train_mask]==label[train_mask]).sum().item/(train_mask.shape[0])
    print("training epoch {} loss {} accuracy {} ".format(step,loss.item(),accuracy))
def eval(model:GCN,loss_function,test_mask,g,label,features,step):
    model.eval()
    with torch.no_grad():
        predict=model(features,g)
        loss=loss_function(predict[test_mask],label[test_mask])
        accuracy=(predict[test_mask]==label[test_mask]).sum().item/(test_mask.shape[0])
        print("training epoch {} loss {} accuracy {} ".format(step,loss.item(),accuracy))
device="cuda:0"
g=g.to(device)
features=features.to(device)
train_mask=train_mask.to(device)
model=GCN()
model=model.to(device)
optim=torch.optim.Adam(model.parameters(),lr=1e-2)
loss_function=torch.nn.CrossEntropyLoss()
for i in range(50):
    train(model,optim,loss_function,train_mask,g,label,features,i)
    eval(model,optim,loss_function,test_mask,g,label,features,i)

