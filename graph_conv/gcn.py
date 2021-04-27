from torch.nn import Module,Linear,ReLU

from dgl import function
gcn_msg=function.copy_u('h','m')
gcn_sum=function.sum('m','h')
class layer(Module):
    def __init__(self,input_params,output_params):
        super().__init__()
        self.linear=Linear(input_params,output_params)
    def forward(self,hidden,g):
        with g.local_scope():
            g.ndata['h']=hidden
            g.update_all(gcn_msg,gcn_sum)#这里认为只需要对临界节点求和即可，因为没有采取attention机制，接下来进行线性变换就可以了
            h=g.ndata['h']
            return self.linear(h)

class GCN(Module):
    def __init__(self):
        super().__init__()
        self.layer=layer(1433,16)
        self.layer2=layer(16,7)
        self.relu=ReLU()
    def forward(self,hidden,g):
        hidden=self.relu(self.layer(hidden,g))
        hidden=self.layer2(hidden,g)
        return hidden
