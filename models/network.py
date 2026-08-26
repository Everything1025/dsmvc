import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Discriminator, FusionLayer, GraphEncoder, InnerProductSimilarity, MLPLearner, SpLearner
from utils.graph_utils import (
    ProbabilitySparsify,
    custom_normalize,
    remove_self_loop,
    sparse_tensor_add_self_loop,
    torch_sparse_to_dgl_graph,
)
class SingleViewModel(nn.Module):
    def __init__(self, temperature, bias,
                 nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse,
                 device):
        super(SingleViewModel, self).__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse)
        self.discriminator = Discriminator(in_dim, hidden_dim, temperature, bias)
        self.fusion_layer = FusionLayer(emb_dim, num_views=2, fusion_type='average')
        self.temperature = temperature
        self.bias = bias
        self.sparse = sparse
        self.device = device
        # newborne
        metric = InnerProductSimilarity()
        processors = [ProbabilitySparsify(temperature=0.05)]
        self.spl = SpLearner(nlayers, in_dim, hidden_dim, nn.ReLU(), 10, True, metric, processors)
        self.graph_learner = MLPLearner(metric, processors, nlayers=2, in_dim=in_dim,
                                        hidden_dim=hidden_dim, activation=nn.ReLU(), sparse=False)
    def forward(self, x, Adj):
        z_specific = self.encoder(x, Adj)
        z_specific = F.normalize(z_specific, dim=1, p=2)
        adj_aug = self.graph_learner(x)
        adj_aug = custom_normalize(adj_aug, 'sym')
        z_aug = self.encoder(x, adj_aug)
        z_aug = F.normalize(z_aug, dim=1, p=2)
        z_view = self.fusion_layer([z_specific, z_aug])
        return adj_aug, z_specific, z_aug, z_view
    def graph_generative_augment(self, adj, view_features, sparse=False):
        if not sparse:
            adj = adj.to_sparse().coalesce()
        adj = remove_self_loop(adj)
        edge_index = adj.indices()
        adj_aug_value = self.discriminator(view_features, edge_index)
        adj_aug = torch.sparse.FloatTensor(edge_index, adj_aug_value, adj.shape).to(adj.device)
        adj_aug = sparse_tensor_add_self_loop(adj_aug)
        adj_aug = custom_normalize(adj_aug, 'sym', sparse=True)
        if sparse:
            adj_aug = torch_sparse_to_dgl_graph(adj_aug)
        else:
            adj_aug = adj_aug.to_dense()
        return adj_aug
class MultiViewModel(nn.Module):
    def __init__(self, nlayers, in_channels, in_dim, hidden_dim, emb_dim, proj_dim, dropout, sparse, temperature, bias,
                 num_g, n_clusters, device):
        super(MultiViewModel, self).__init__()
        self.views = nn.ModuleList([
            SingleViewModel(temperature, bias,
                            nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse,
                            device)
            for in_channel in in_channels
        ])
        self.aligners = nn.ModuleList()
        for in_channel in in_channels:
            self.aligners.append(nn.Linear(in_channel, in_dim))
        self.aligners_out = nn.ModuleList()
        for in_channel in in_channels:
            self.aligners_out.append(nn.Linear(in_dim, in_channel))
        self.dropout = dropout
        self.decoders = nn.ModuleList()
        for in_channel in in_channels:
            self.decoders.append(nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, in_channel)
            ))
        self.device = device
        self.num_g = num_g
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, emb_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1.0
        self.fusion_layer = FusionLayer(in_channel=emb_dim, num_views=len(in_channels), fusion_type='average')
        self.proj_s = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])
        self.proj_u = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])
        self.proj_f = nn.ModuleList([nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        ) for _ in range(self.num_g)])
    def forward(self, x, adjs):
        z_specific_adjs = []
        z_aug_adjs = []
        z_views = []
        aug_adjs = []
        x_align = []
        for i in range(len(adjs)):
            x_align.append(F.dropout(self.aligners[i](x[i]), p=self.dropout, training=self.training))
            adj_aug, z_specific, z_aug, z_view = self.views[i](x_align[i], adjs[i])
            z_specific_adjs.append(z_specific)
            z_aug_adjs.append(z_aug)
            z_views.append(z_view)
            aug_adjs.append(adj_aug)
        z_proj_s = [self.proj_s[i](z_specific_adjs[i]) for i in range(self.num_g)]
        z_proj_u = [self.proj_u[i](z_aug_adjs[i]) for i in range(self.num_g)]
        z_proj_a = [self.proj_f[i](z_specific_adjs[i] + z_aug_adjs[i]) for i in range(self.num_g)]
        z_combined = self.fusion_layer(z_views)
        z_combined = F.normalize(z_combined, dim=1, p=2)

        return adjs, aug_adjs, z_specific_adjs,  z_combined, z_proj_s, z_proj_u, z_proj_a
    def compute_cluster_probs(self, z_views, z_combined, adjs):
        qs = []
        q = 1.0 / (1.0 + torch.sum(torch.pow((z_combined).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        for i in range(len(adjs)):
            q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_views[i].unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q1 = q1.pow((self.v + 1.0) / 2.0)
            q1 = (q1.t() / torch.sum(q1, 1)).t()
            qs.append(q1)
        return q, qs
