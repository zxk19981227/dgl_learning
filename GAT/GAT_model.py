import dgl
from torch.nn import functional as F
from torch.nn import Module,Linear
import torch
class GAT_layer(Module):
    def __init__(self,g,input_dim,output_dim):
        super().__init__()
        self.linear=Linear(input_dim,output_dim)
        self.g=g
        self.attention=Linear(2*output_dim,1)
    def message_function(self,edges):
        current_edge=torch.cat([edges.src['h'],edges.dst['h']],1)
        current_edge=self.attention(current_edge)
        return {'e':F.leaky_relu(current_edge)}
    def apply_edge(self,edges):
        return {'z':edges.src['h'],'e':edges.data['e']}
    def apply_node(self,nodes):
        weight=nodes.mailbox['e']
        weight=F.softmax(weight,dim=1)*nodes.mailbox['z']
        return {'h':torch.sum(weight,1)}
    def forward(self,hidden):
        self.g.ndata['h']=self.linear(hidden)
        self.g.apply_edges(self.message_function)
        self.g.update_all(self.apply_edge,self.apply_node)
        return self.g.ndata['h']
class GAT(Module):
    def __init__(self,g,input_dim,output_dim,layers):
        super().__init__()
        self.layers=torch.nn.ModuleList()
        self.layers.append(GAT_layer(g,input_dim,8))
        self.linear=Linear(8,output_dim)
        for layer in range(layers-1):
            self.layers.append(GAT_layer(g,8,8))
    def forward(self,hidden):
        for layer in self.layers:
            hidden=layer(hidden)
            hidden=F.relu(hidden)
        return hidden

