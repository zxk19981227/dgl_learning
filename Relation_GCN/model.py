from torch.nn import Module
import torch
import dgl.function as f
from torch.nn.functional import relu
class R_GCN_Layer(Module):
    def __init__(self,input_dim,output_dim,rel_num,activation=None,is_input=False):
        super().__init__()
        self.is_input=is_input
        self.weight=torch.nn.Parameter(torch.Tensor(rel_num,input_dim,output_dim))#this shape means each type relation should have a extract parameters
        torch.nn.init.xavier_uniform_(self.weight)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.rel_num=rel_num
        self.activation=activation
    def forward(self,g):
        if self.is_input:
            def message_func(edges):
                print(self.weight.shape)
                weight=self.weight
                weight=weight.view(self.input_dim*self.rel_num,self.output_dim)
                # print(edges.src['id'])
                # print(weight.shape)
                relation_index=edges.data['rel_type']*self.input_dim+edges.src['id']# because the shape is gatherd by every edge type, so id is this
                #why this operation works because the init input is one-hot, which means does nothing (mentioned in paper)
                return {'msg':weight[relation_index]*edges.data['norm']}
        else:
            def message_func(edges):
                weight=self.weight[edges.data['rel_type']]
                features=torch.bmm(edges.src['h'].unsqueeze(1),weight).squeeze()
                return {'msg':features*edges.data['norm']}
        def apply_node(nodes):
            # print("apply_node")
            h=nodes.data['h']
            if self.activation:
                h=self.activation(h)
            return {'h':h}
        g.update_all(message_func,f.sum('msg','h'),apply_node)
class Model(Module):
    def __init__(self,hidden_layer_num,rel_num,node_num):
        self.dim=16
        self.rel_num=rel_num
        super().__init__()
        self.n_feature=torch.arange(node_num)
        self.layers=torch.nn.ModuleList()
        self.node_num=node_num
        self.predict_num=4
        self.layers.append(R_GCN_Layer(self.node_num,self.dim,rel_num,activation=relu,is_input=True))
        for i in range(hidden_layer_num):
            self.layers.append(R_GCN_Layer(self.dim,self.dim,rel_num,activation=relu))
        self.layers.append(R_GCN_Layer(self.dim,self.predict_num,rel_num))
    def forward(self,g):
        g.ndata['id']=self.n_feature
        for layer in self.layers:
            layer(g)





