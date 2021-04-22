from torch.nn import Module
import torch
import dgl.function as f
class R_GCN_Layer(Module):
    def __init__(self,input_dim,output_dim,rel_num,activation=None):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.Tensor(rel_num,input_dim,output_dim))#this shape means each type relation should have a extract parameters
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.rel_num=rel_num
        self.activation=activation
    def forward(self,g,is_input):
        if is_input:
            def message_func(edges):
                weight=self.weight
                weight=weight.view(self.input_dim*self.rel_num,self.output_dim)
                relation_index=edges.data['rel_type']*self.input_dim+edges.src['id']# because the shape is gatherd by every edge type, so id is this
                #why this operation works because the init input is one-hot, which means does nothing (mentioned in paper)
                print(edges.data['norm'])
                return {'msg':weight[relation_index]*edges.data['norm']}
        else:
            def message_func(edges):
                weight=self.weight[edges.data['rel_type']]
                features=torch.bmm(weight,edges.src['h'])
                return {'h':features*edges.data['norm']}
        def apply_node(nodes):
            h=nodes.data['h']
            if self.activation:
                h=self.activation('h')
            return h
        g.update_all(message_func,f.sum('msg','h'),apply_node)
class Model(Module):
    def __init__(self,hidden_layer_num,rel_num):
        self.dim=16
        self.rel_num=rel_num
        super().__init__()
        self.layers=torch.nn.ModuleList()
        self.layers.append(R_GCN_Layer(self.dim,self.dim,rel_num=rel_num,activation=True))
        for i in range(hidden_layer_num):
            self.layers.append(R_GCN_Layer(self.dim,self.dim,rel_num))
        self.layers.append(R_GCN_Layer(self.dim,rel_num,rel_num))





