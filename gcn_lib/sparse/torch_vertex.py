import torch
from torch import nn, Tensor
from .torch_nn import MLP

from typing import Callable
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor
from gcn_lib.sparse import MultiSeq, MLP


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class AttrRelativeEdgeConvGlobalPool2(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(AttrRelativeEdgeConvGlobalPool2,
              self).__init__(aggr='mean', **kwargs)
        self.nn = MLP([in_channels * 2 + 4, out_channels,
                      out_channels], 'relu', 'batch')
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=True)
        self.mlp_node = MLP([in_channels, out_channels], 'relu', 'batch')

        self.in_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, x_node, edge_index, edge_weight=None, edge_attr=None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(
            edge_index, x=x, norm=edge_weight, attr=edge_attr, size=None)
        out += self.lin_r(x[1])
        x_node = self.mlp_node(x_node)

        return out, x_node

    def message(self, x_i: Tensor, x_j: Tensor, norm, attr) -> Tensor:
        f = torch.cat([x_i, x_j - x_i, attr], dim=1)
        # f = torch.cat([x_j, attr], dim = 1)

        if norm is None:
            return self.nn(f)
        else:
            return norm.view(-1, 1) * self.nn(f)
        # return self.nn(x_j - x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class AttrRelativeEdgeConvGlobalPool(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, in_channels, out_channels, **kwargs):
        super(AttrRelativeEdgeConvGlobalPool,
              self).__init__(aggr='mean', **kwargs)
        self.nn = nn
        self.mlp = MultiSeq(*[MLP([in_channels, out_channels]),
                              ])
        # self.mlp_attr = MultiSeq(*[MLP([4, out_channels]),
        # ])

        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        # self.lin_l = torch.nn.Linear(out_channels, out_channels, bias=True)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=True)

        self.in_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(
            edge_index, x=x, norm=edge_weight, attr=edge_attr, size=None)
        # out = self.lin_l(out)

        x_r = x[1][:, 0:self.in_channels]
        out += self.lin_r(x_r)

        # out += self.mlp(x[1][:, self.in_channels:] - x_r)
        out += self.mlp(x[1][:, self.in_channels:])

        return out

    def message(self, x_i: Tensor, x_j: Tensor, norm, attr) -> Tensor:
        # return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))
        # return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))
        '''
        diff = x_j - x_i
        euc_d = torch.norm(diff, dim =)
        angle = diff / (np.sqrt(euc_d2) + 1e-7)
        w = 1 / np.exp(euc_d2)
        '''
        x_i_root = x_i[:, self.in_channels:]
        x_i = x_i[:, 0:self.in_channels]
        x_j = x_j[:, 0:self.in_channels]

        # f = torch.cat([x_j - x_i, x_i_root - x_i, attr], dim = 1)
        f = torch.cat([x_i, x_j - x_i, attr], dim=1)
        # f = torch.cat([x_j - x_i, attr], dim = 1)

        if norm is None:
            # return self.nn(torch.cat([x_j - x_i, x_i, attr], dim = 1))
            # return self.nn(x_j - x_i) + 0.1 * self.mlp_attr(attr)
            return self.nn(f)
        else:
            # return norm.view(-1, 1) * self.nn(torch.cat([x_j - x_i, x_i, attr], dim = 1))
            return norm.view(-1, 1) * self.nn(f)
        # return self.nn(x_j - x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class EdgConvGlobalPool(AttrRelativeEdgeConvGlobalPool):
    """
    Edge convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
        # super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)
        # super(EdgConv, self).__init__(torch.nn.Linear(in_channels, out_channels), in_channels, out_channels, aggr)
        # super(EdgConv, self).__init__(MLP([in_channels * 2, out_channels], act, norm, bias), in_channels, out_channels, aggr)
        super(EdgConvGlobalPool, self).__init__(MLP(
            [in_channels * 2 + 4, out_channels], act, norm, bias), in_channels, out_channels)
        # super(EdgConvGlobalPool, self).__init__(MLP([in_channels + 4, out_channels], act, norm, bias), in_channels, out_channels)
        # super(EdgConvGlobalPool, self).__init__(MLP([in_channels + 4, out_channels, out_channels], act, norm, bias), in_channels, out_channels)
        # super(EdgConvGlobalPool, self).__init__(MLP([in_channels, out_channels], act, norm, bias), in_channels, out_channels)
        # super(AttrEdgConv, self).__init__(MLP([in_channels + 6, out_channels], act, norm, bias), in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        return super(EdgConvGlobalPool, self).forward(x, edge_index, edge_weight, edge_attr)


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='gcn',  # conv='edge',
                 act='relu', norm=None, bias=True, heads=8):
        super(GraphConv, self).__init__()
        self.conv = conv.lower()
        # elif conv.lower() == 'multilayer_edge':
        #     self.gconv = MultilayerEdgConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'attr_edge':
        #     self.gconv = AttrEdgConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'attr_edge_cf':
        #     self.gconv = AttrEdgeConvCF(in_channels, out_channels, act, norm, bias)
        if conv.lower() == 'attr_edge_gp':
            self.gconv = EdgConvGlobalPool(
                in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'edge':
        #     self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'attr_edge_gp2':
            self.gconv = AttrRelativeEdgeConvGlobalPool2(
                in_channels, out_channels)
        # elif conv.lower() == 'mr':
        #     self.gconv = MRConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'gat':
        #     self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        # elif conv.lower() == 'gcn':
        #     self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'gin':
        #     self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'sage':
        #     self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, False)
        # elif conv.lower() == 'rsage':
        #     self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, True)
        else:
            raise NotImplementedError(
                'conv {} is not implemented'.format(conv))

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, x_node=None):
        if self.conv == 'attr_edge' or self.conv == 'multilayer_edge' or self.conv == 'attr_edge_gp':
            return self.gconv(x, edge_index, edge_weight, edge_attr)
        elif self.conv == 'attr_edge_cf':
            return self.gconv(x, edge_index, edge_weight, edge_attr, pos)
        elif self.conv == 'edge' and edge_weight is not None:
            return self.gconv(x, edge_index, edge_weight)
        if self.conv == 'attr_edge_gp2':
            return self.gconv(x, x_node, edge_index, edge_weight, edge_attr)
        else:
            return self.gconv(x, edge_index)


class ResBlock(nn.Module):
    """
    Residual graph convolution block
    """

    def __init__(self, channels, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv,
                              act, norm, bias, **kwargs)
        self.res_scale = res_scale
        self.channels = channels

    def forward(self, x, edge, edge_weight=None, edge_attr=None, pos=None, x_node=None):
        if isinstance(self.body.gconv, EdgConvGlobalPool):
            return self.body(x, edge, edge_weight, edge_attr, pos) + x[:, 0:self.channels]*self.res_scale
        elif isinstance(self.body.gconv, AttrRelativeEdgeConvGlobalPool2):
            out, out_node = self.body(
                x, edge, edge_weight, edge_attr, x_node=x_node)
            # out += x * self.res_scale
            # out_node += x_node * self.res_scale
            return out, out_node
        else:
            return self.body(x, edge, edge_weight, edge_attr, pos) + x*self.res_scale

# class MRConv(nn.Module):
#     """
#     Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
#         super(MRConv, self).__init__()
#         self.nn = MLP([in_channels*2, out_channels], act, norm, bias)
#         self.aggr = aggr

#     def forward(self, x, edge_index):
#         """"""
#         x_j = tg.utils.scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
#         return self.nn(torch.cat([x, x_j], dim=1))

# class AttrEdgeConvCF(MessagePassing):
#     r"""The edge convolutional operator from the `"Dynamic Graph CNN for
#     Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
#     .. math::
#         \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
#         h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
#         \mathbf{x}_j - \mathbf{x}_i),
#     where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps pair-wise concatenated node features :obj:`x` of shape
#             :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
#             *e.g.*, defined by :class:`torch.nn.Sequential`.
#         aggr (string, optional): The aggregation scheme to use
#             (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#             (default: :obj:`"max"`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, in_channels, out_channels, act, norm, bias):
#         super(AttrEdgeConvCF, self).__init__(aggr='mean')
#         self.mlps = torch.nn.ModuleList()
#         for i in range(8):
#             self.mlps.append(MLP([in_channels + 4, out_channels, out_channels], act, norm, bias))

#         #self.lin_l = torch.nn.Linear(out_channels, out_channels, bias=True)
#         self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=True)

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.bn = torch.nn.BatchNorm1d(out_channels)

#         self.reset_parameters()


#     def reset_parameters(self):
#         #reset(self.nn)
#         pass


#     def forward(self, x, edge_index, edge_weight = None, edge_attr = None, pos = None) -> Tensor:
#         """"""
#         x = torch.cat([pos, x], dim = 1)
#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)

#         out = self.propagate(edge_index, x=x, norm = edge_weight, attr = edge_attr, size=None)
#         #out = self.lin_l(out)

#         x_r = x[1][:, 2:]
#         #print(x_r.size(), self.in_channels)
#         out += self.lin_r(x_r)
#         # propagate_type: (x: PairTensor)
#         return out

#     def message(self, x_i: Tensor, x_j: Tensor, norm, attr) -> Tensor:
#         pos_i = x_i[:, 0:2]
#         pos_j = x_j[:, 0:2]

#         diff = pos_j - pos_i
#         con0 = torch.sign(diff[:, 0])
#         con1 = torch.sign(diff[:, 1])
#         con2 = torch.sign(torch.abs(diff[:, 0]) - torch.abs(diff[:, 1]))
#         con0[con0 < 0] = 0
#         con1[con1 < 0] = 0
#         con2[con2 < 0] = 0
#         idx = con0 + con1 * 2 + con2 * 4
#         #print(idx)

#         tan = diff[:, 1] / diff[:, 0]

#         x_i = x_i[:, 2:]
#         x_j = x_j[:, 2:]

#         f_list = []
#         for i in range(8):
#             f_list.append(self.mlps[i](torch.cat([x_j - x_i, attr], dim = 1)).unsqueeze(1))
#         f = torch.cat(f_list, dim = 1)
#         #print(f.size(), idx.long(), idx.size())
#         idx0 = torch.LongTensor(list(range(f.size(0))))
#         f = f[idx0, idx.long()]
#         #print(f.size())
#         #raise SystemExit
#         if norm is None:
#             return f
#         else:
#             return norm.view(-1, 1) * f
#         #return self.nn(x_j - x_i)

#     def __repr__(self):
#         return '{}'.format(self.__class__.__name__)

# class AttrRelativeEdgeConv(MessagePassing):
#     r"""The edge convolutional operator from the `"Dynamic Graph CNN for
#     Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
#     .. math::
#         \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
#         h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
#         \mathbf{x}_j - \mathbf{x}_i),
#     where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps pair-wise concatenated node features :obj:`x` of shape
#             :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
#             *e.g.*, defined by :class:`torch.nn.Sequential`.
#         aggr (string, optional): The aggregation scheme to use
#             (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#             (default: :obj:`"max"`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, nn: Callable, in_channels, out_channels, **kwargs):
#         super(AttrRelativeEdgeConv, self).__init__(aggr='mean', **kwargs)
#         self.nn = nn
#         self.mlp = MultiSeq(*[MLP([in_channels, 64]),
#             MLP([64, in_channels]),
#         ])

#         #self.lin_l = torch.nn.Linear(out_channels, out_channels, bias=True)
#         self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=True)
#         self.reset_parameters()


#     def reset_parameters(self):
#         reset(self.nn)

#     def forward(self, x, edge_index, edge_weight = None, edge_attr = None) -> Tensor:
#         """"""
#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)

#         out = self.propagate(edge_index, x=x, norm = edge_weight, attr = edge_attr, size=None)
#         #out = self.lin_l(out)

#         x_r = x[1]
#         out += self.lin_r(x_r)
#         # propagate_type: (x: PairTensor)
#         return out

#     def message(self, x_i: Tensor, x_j: Tensor, norm, attr) -> Tensor:
#         #return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))
#         #return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))

#         '''
#         diff = x_j - x_i
#         euc_d = torch.norm(diff, dim =)
#         angle = diff / (np.sqrt(euc_d2) + 1e-7)
#         w = 1 / np.exp(euc_d2)
#         '''

#         if norm is None:
#             #return self.nn(torch.cat([x_j - x_i, x_i, attr], dim = 1))
#             return self.nn(torch.cat([x_j - x_i, attr], dim = 1))
#         else:
#             #return norm.view(-1, 1) * self.nn(torch.cat([x_j - x_i, x_i, attr], dim = 1))
#             return norm.view(-1, 1) * self.nn(torch.cat([x_j - x_i, attr], dim = 1))
#         #return self.nn(x_j - x_i)

#     def __repr__(self):
#         return '{}(nn={})'.format(self.__class__.__name__, self.nn)

# class WeightedRelativeEdgeConv(MessagePassing):
#     r"""The edge convolutional operator from the `"Dynamic Graph CNN for
#     Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
#     .. math::
#         \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
#         h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
#         \mathbf{x}_j - \mathbf{x}_i),
#     where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps pair-wise concatenated node features :obj:`x` of shape
#             :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
#             *e.g.*, defined by :class:`torch.nn.Sequential`.
#         aggr (string, optional): The aggregation scheme to use
#             (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
#             (default: :obj:`"max"`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, nn: Callable, in_channels, out_channels, **kwargs):
#         super(WeightedRelativeEdgeConv, self).__init__(aggr='mean', **kwargs)
#         self.nn = nn
#         self.mlp = MultiSeq(*[MLP([in_channels, 64]),
#             MLP([64, in_channels]),
#         ])

#         self.lin_l = torch.nn.Linear(out_channels, out_channels, bias=True)
#         self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=True)
#         self.reset_parameters()


#     def reset_parameters(self):
#         reset(self.nn)

#     def forward(self, x, edge_index, edge_weight = None) -> Tensor:
#         """"""
#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)

#         out = self.propagate(edge_index, x=x, norm = edge_weight, size=None)
#         #out = self.lin_l(out)

#         x_r = x[1]
#         out += self.lin_r(x_r)
#         # propagate_type: (x: PairTensor)
#         return out

#     def message(self, x_i: Tensor, x_j: Tensor, norm) -> Tensor:
#         #return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))
#         #return self.nn(torch.cat([x_j - x_i + self.mlp(x_i), x_i], dim = 1))
#         if norm is None:
#             return self.nn(torch.cat([x_j - x_i, x_i], dim = 1))
#         else:
#             return norm.view(-1, 1) * self.nn(torch.cat([x_j - x_i, x_i], dim = 1))
#         #return self.nn(x_j - x_i)

#     def __repr__(self):
#         return '{}(nn={})'.format(self.__class__.__name__, self.nn)

# class EdgConv(tg.nn.EdgeConv):
# class EdgConv(WeightedRelativeEdgeConv):
#     """
#     Edge convolution layer (with activation, batch normalization)
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
#         #super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)
#         #super(EdgConv, self).__init__(torch.nn.Linear(in_channels, out_channels), in_channels, out_channels, aggr)
#         #super(EdgConv, self).__init__(MLP([in_channels * 2, out_channels], act, norm, bias), in_channels, out_channels, aggr)
#         super(EdgConv, self).__init__(MLP([in_channels * 2, out_channels], act, norm, bias), in_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight = None):
#         return super(EdgConv, self).forward(x, edge_index, edge_weight)

# class AttrEdgConv(AttrRelativeEdgeConv):
#     """
#     Edge convolution layer (with activation, batch normalization)
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
#         #super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)
#         #super(EdgConv, self).__init__(torch.nn.Linear(in_channels, out_channels), in_channels, out_channels, aggr)
#         #super(EdgConv, self).__init__(MLP([in_channels * 2, out_channels], act, norm, bias), in_channels, out_channels, aggr)
#         #super(AttrEdgConv, self).__init__(MLP([in_channels * 2 + 4, out_channels], act, norm, bias), in_channels, out_channels)
#         super(AttrEdgConv, self).__init__(MLP([in_channels+ 4, out_channels], act, norm, bias), in_channels, out_channels)
#         #super(AttrEdgConv, self).__init__(MLP([in_channels + 6, out_channels], act, norm, bias), in_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight = None, edge_attr = None):
#         return super(AttrEdgConv, self).forward(x, edge_index, edge_weight, edge_attr)

# class MultilayerEdgConv(AttrRelativeEdgeConv):
#     """
#     Edge convolution layer (with activation, batch normalization)
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
#         #super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)
#         #super(EdgConv, self).__init__(torch.nn.Linear(in_channels, out_channels), in_channels, out_channels, aggr)
#         #super(EdgConv, self).__init__(MLP([in_channels * 2, out_channels], act, norm, bias), in_channels, out_channels, aggr)
#         super(MultilayerEdgConv, self).__init__(MLP([in_channels + 4, out_channels, out_channels],
#         act, norm, bias), in_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight = None, edge_attr = None):
#         return super(MultilayerEdgConv, self).forward(x, edge_index, edge_weight, edge_attr)

# class GATConv(nn.Module):
#     """
#     Graph Attention Convolution layer (with activation, batch normalization)
#     """
#     def __init__(self, in_channels, out_channels,  act='relu', norm=None, bias=True, heads=8):
#         super(GATConv, self).__init__()
#         self.gconv = tg.nn.GATConv(in_channels, out_channels, heads, bias=bias)
#         m =[]
#         if act:
#             m.append(act_layer(act))
#         if norm:
#             m.append(norm_layer(norm, out_channels * heads))
#         self.unlinear = nn.Sequential(*m)

#     def forward(self, x, edge_index):
#         out = self.unlinear(self.gconv(x, edge_index))
#         return out

# class SAGEConv(tg.nn.SAGEConv):
#     r"""The GraphSAGE operator from the `"Inductive Representation Learning on
#     Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

#     .. math::
#         \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
#         \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

#         \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
#         {\| \mathbf{\hat{x}}_i \|_2}.

#     Args:
#         in_channels (int): Size of each input sample.
#         out_channels (int): Size of each output sample.
#         normalize (bool, optional): If set to :obj:`False`, output features
#             will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  nn,
#                  norm=True,
#                  bias=True,
#                  relative=False,
#                  **kwargs):
#         self.relative = relative
#         if norm is not None:
#             super(SAGEConv, self).__init__(in_channels, out_channels, True, bias, **kwargs)
#         else:
#             super(SAGEConv, self).__init__(in_channels, out_channels, False, bias, **kwargs)
#         self.nn = nn

#     def forward(self, x, edge_index, size=None):
#         """"""
#         if size is None:
#             edge_index, _ = remove_self_loops(edge_index)
#             edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         x = x.unsqueeze(-1) if x.dim() == 1 else x
#         return self.propagate(edge_index, size=size, x=x)

#     def message(self, x_i, x_j):
#         if self.relative:
#             x = torch.matmul(x_j - x_i, self.weight)
#         else:
#             x = torch.matmul(x_j, self.weight)
#         return x

#     def update(self, aggr_out, x):
#         out = self.nn(torch.cat((x, aggr_out), dim=1))
#         if self.bias is not None:
#             out = out + self.bias
#         if self.normalize:
#             out = F.normalize(out, p=2, dim=-1)
#         return out

# class RSAGEConv(SAGEConv):
#     """
#     Residual SAGE convolution layer (with activation, batch normalization)
#     """

#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, relative=False):
#         nn = MLP([out_channels + in_channels, out_channels], act, norm, bias)
#         super(RSAGEConv, self).__init__(in_channels, out_channels, nn, norm, bias, relative)


# class SemiGCNConv(nn.Module):
#     """
#     SemiGCN convolution layer (with activation, batch normalization)
#     """

#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
#         super(SemiGCNConv, self).__init__()
#         self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
#         m = []
#         if act:
#             m.append(act_layer(act))
#         if norm:
#             m.append(norm_layer(norm, out_channels))
#         self.unlinear = nn.Sequential(*m)

#     def forward(self, x, edge_index):
#         out = self.unlinear(self.gconv(x, edge_index))
#         return out

# class GinConv(tg.nn.GINConv):
#     """
#     GINConv layer (with activation, batch normalization)
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
#         super(GinConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

#     def forward(self, x, edge_index):
#         return super(GinConv, self).forward(x, edge_index)
