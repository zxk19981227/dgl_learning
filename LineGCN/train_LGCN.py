from numpy import mean
from torch.nn.functional import cross_entropy
import dgl
import torch
from LineGCN.model import model
from torch.utils.data import DataLoader
train_dataset=dgl.data.CoraBinary()
train_set=DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, drop_last=True)

model=model(2,1,2)
optim=torch.optim.Adam(model.parameters(),lr=1e-2)
for i in range(20):
    total_loss=[]
    cor=0
    acc=0
    for [g,pmpd,label] in train_set:
        lg = g.line_graph(backtracking=False)
        predict=model(g,lg,pmpd)
        reverse_label=1-label
        loss=min(cross_entropy(predict,torch.tensor(label)),cross_entropy(predict,torch.tensor(reverse_label)))
        loss.backward()
        optim.step()
        optim.zero_grad()
        predict=torch.argmax(predict,-1)
        acc+=(predict==torch.tensor(label)).int().sum()
        cor+=label.shape[0]
        total_loss.append(loss.item())
    print("epoch {} accuracy {} loss {}".format(i,acc/cor,mean(total_loss)))
