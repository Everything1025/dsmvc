import dgl
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from config import EOS

def get_sparse_diag(adj):
    adj = adj.coalesce()
    diag_index = torch.nonzero(adj.indices()[0] == adj.indices()[1]).flatten()
    values = adj.values()[diag_index]
    indices = adj.indices()[:, diag_index]
    return indices, values
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph
def remove_self_loop(adj):
    adj = adj.coalesce()
    non_diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
    adj = torch.sparse.FloatTensor(adj.indices()[:, non_diag_index], adj.values()[non_diag_index],
                                   adj.shape).coalesce()
    return adj
def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)
    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1),
                                       torch.cat((values, adj.values()), dim=0), adj.shape)
    return adj_new.coalesce()
def custom_normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value
        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()

def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    norm_row = torch.zeros(X.shape[0]).to(X.device)
    norm_col = torch.zeros(X.shape[0]).to(X.device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index: index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1): (end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1): (end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1): (end) * (k + 1)] = (
            torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        )
        norm_row[index:end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5)
    return rows, cols, values

class KNNSparsify:
    def __init__(self, k, discrete=False, self_loop=True):
        super(KNNSparsify, self).__init__()
        self.k = k
        self.discrete = discrete
        self.self_loop = self_loop
    def __call__(self, adj):
        _, indices = adj.topk(k=int(self.k + 1), dim=-1)
        assert torch.max(indices) < adj.shape[1]
        mask = torch.zeros(adj.shape).to(adj.device)
        mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.
        mask.requires_grad = False
        if self.discrete:
            sparse_adj = mask.to(torch.float)
        else:
            sparse_adj = adj * mask
        if not self.self_loop:
            sparse_adj.fill_diagonal_(0)
        return sparse_adj
class ThresholdSparsify:
    def __init__(self, threshold):
        super(ThresholdSparsify, self).__init__()
        self.threshold = threshold
    def __call__(self, adj):
        return torch.where(adj < self.threshold, torch.zeros_like(adj), adj)
class ProbabilitySparsify:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
    def __call__(self, prob):
        prob = torch.clamp(prob, 0.01, 0.99)
        adj = RelaxedBernoulli(temperature=torch.Tensor([self.temperature]).to(prob.device),
                               probs=prob).rsample()
        eps = 0.5
        mask = (adj > eps).detach().float()
        adj = adj * mask + 0.0 * (1 - mask)
        return adj
class Discretize:
    def __init__(self):
        super(Discretize, self).__init__()
    def __call__(self, adj):
        adj[adj != 0] = 1.0
        return adj
class AddEye:
    def __init__(self):
        super(AddEye, self).__init__()
    def __call__(self, adj):
        adj += torch.eye(adj.shape[0]).to(adj.device)
        return adj
class LinearTransform:
    def __init__(self, alpha):
        super(LinearTransform, self).__init__()
        self.alpha = alpha
    def __call__(self, adj):
        adj = adj * self.alpha - self.alpha
        return adj
class NonLinearize:
    def __init__(self, non_linearity='relu', alpha=1.0):
        super(NonLinearize, self).__init__()
        self.non_linearity = non_linearity
        self.alpha = alpha
    def __call__(self, adj):
        if self.non_linearity == 'elu':
            return F.elu(adj) + 1
        elif self.non_linearity == 'relu':
            return F.relu(adj)
        elif self.non_linearity == 'none':
            return adj
        else:
            raise NameError('We dont support the non-linearity yet')
class Symmetrize:
    def __init__(self):
        super(Symmetrize, self).__init__()
    def __call__(self, adj):
        return (adj + adj.T) / 2
class Normalize:
    def __init__(self, mode='sym', eos=1e-10):
        super(Normalize, self).__init__()
        self.mode = mode
        self.EOS = eos
    def __call__(self, adj):
        if self.mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + self.EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif self.mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + self.EOS)
            return inv_degree[:, None] * adj
        elif self.mode == "row_softmax":
            return F.softmax(adj, dim=1)
        elif self.mode == "row_softmax_sparse":
            return torch.sparse.softmax(adj.to_sparse(), dim=1).to_dense()
        else:
            raise Exception('We dont support the normalization mode')

def delete_graph_all(labels, adjs):
    device = labels.device
    labels = labels.detach()
    torch.unique(labels)
    refined_adjs = []
    for v, adj in enumerate(adjs):
        torch.zeros_like(adj, device=device)
        label_mask = (labels[:, None] == labels[None, :]).float()
        adj_matrix = adj * label_mask
        refined_adjs.append(adj_matrix)
    return refined_adjs
