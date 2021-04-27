import dgl.function as fn
from torch.nn import Module, Linear,BatchNorm1d
import torch
from torch.nn.functional import relu


def aggrate(radius,g,hidden):
    result=[]
    g.ndata['tmp'] = hidden
    # gather the one-hop information from graph
    # this actions aim at adding 1 to the total combination of 2^n
    g.update_all(fn.copy_u('tmp', 'm'), fn.sum('m', 'tmp'))
    result.append(g.ndata['tmp'])
    for i in range(radius - 1):  # this because zero has been applied and total power is 1+1+2+...2^n=2^(n+1)
        for j in range(2 ** i):
            # I thought if there could be accerlated through squraing exponentiating
            g.update_all(fn.copy_u('tmp', 'm'), fn.sum('m', 'tmp'))
        result.append(g.ndata['tmp'])
    return result


class LGCN_Layer(Module):
    def __init__(self, radius, input_dim, output_dim):
        super().__init__()
        self.prev_g_linear = Linear(input_dim, output_dim)
        self.prev_lg_linear = Linear(input_dim, output_dim)
        self.deg_g_linear = Linear(input_dim, output_dim)
        self.deg_lg_linear = Linear(input_dim, output_dim)
        self.g_fuse = Linear(input_dim, output_dim)
        self.lg_fuse = Linear(input_dim, output_dim)
        # self.orig_radius = radius_Layer(radius)
        # self.line_radius = radius_Layer(radius)
        self.ba=BatchNorm1d(output_dim)
        self.lin=BatchNorm1d(output_dim)
        self.radius = radius
        self.g_linear = [Linear(input_dim, output_dim) for i in range(radius)]
        self.lg_linear = [Linear(input_dim, output_dim) for i in range(radius)]

    def forward(self, g, lg, g_feature, lg_feature, g_degree, lg_degree, pm_pd):
        orig_result=aggrate(self.radius,g, g_feature)
        line_result=aggrate(self.radius,lg, lg_feature)
        g_radius_feature = [line(feature) for line, feature in zip(self.g_linear, orig_result)]
        lg_radius_feature = [line(feature) for line, feature in zip(self.lg_linear, line_result)]
        sum_g_feature = sum(g_radius_feature)
        sum_lg_feature = sum(lg_radius_feature)
        orig_feature = self.prev_g_linear(g_feature)
        line_feature = self.prev_lg_linear(lg_feature)
        g_fuse_feature = self.g_fuse(g_degree * g_feature)
        lg_fuse_feature = self.lg_fuse(lg_degree * lg_feature)
        g_deg_feature = self.deg_g_linear(torch.mm(pm_pd, lg_feature))
        lg_deg_feature = self.deg_lg_linear(torch.mm(torch.transpose(pm_pd,0,1), g_feature))
        g_feature = sum_g_feature + orig_feature + g_deg_feature + g_fuse_feature
        lg_feature = sum_lg_feature + line_feature + lg_deg_feature + lg_fuse_feature
        feature_size = g_feature.shape[-1]
        result_feature = torch.cat([relu(g_feature[:feature_size // 2]), g_feature[feature_size // 2:]])
        result_lg_feature = torch.cat([relu(lg_feature[:feature_size // 2]), lg_feature[feature_size // 2:]])
        return self.ba(result_feature), self.lin(result_lg_feature)


class model(Module):
    def __init__(self, hidden_num, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(LGCN_Layer(3, input_dim, 16))
        for i in range(hidden_num):
            self.layers.append(LGCN_Layer(3, 16, 16))
        self.linear = Linear(16, output_dim)

    def sparse2th(self,mat):
        value = mat.data
        indices = torch.LongTensor([mat.row, mat.col])
        tensor = torch.sparse.FloatTensor(indices, torch.from_numpy(value).float(), mat.shape)
        return tensor
    def forward(self, g,lg, pm_pd):

        pm_pd=self.sparse2th(pm_pd)
        g_init = g.in_degrees().float().unsqueeze(1)
        g_degrees = g_init
        lg_init = lg.in_degrees().float().unsqueeze(1)
        lg_degree = lg_init
        for layer in self.layers:
            g_init, lg_init = layer(g, lg, g_init, lg_init, g_degrees, lg_degree, pm_pd)
        return self.linear(g_init)
