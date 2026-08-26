import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import eps
from utils.graph_utils import knn_fast

class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def init_para(self):
        self.linear.reset_parameters()
    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output
class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']
class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, dropout, sparse):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.gnn_encoder_layers = nn.ModuleList()
        self.act = nn.ReLU()
        # self.act = nn.ELU()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))
        self.sparse = sparse
    def forward(self, x, Adj):
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        return x
class FusionLayer(nn.Module):
    def __init__(self, in_channel, num_views, fusion_type='average'):
        """
        :param fusion_type: include concatenate/average
        """
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views
        self.pai_fea = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        self.pai_adj = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)
        self.attention = Attention(in_channel)
    def forward(self, features):
        if self.fusion_type == "concat":
            combined_feature = torch.cat(features, dim=1)
        elif self.fusion_type == "average":
            features = torch.stack(features, dim=1)
            combined_feature = torch.sum(features, dim=1)
        elif self.fusion_type == "weighted":
            exp_sum_pai_fea = 0
            for i in range(self.num_views):
                exp_sum_pai_fea += torch.exp(self.pai_fea[i])
            combined_feature = (torch.exp(self.pai_fea[0]) / exp_sum_pai_fea) * features[0]
            for i in range(1, self.num_views):
                combined_feature = combined_feature + (torch.exp(self.pai_fea[i]) / exp_sum_pai_fea) * features[i]
        elif self.fusion_type == "attention":
            features = torch.stack(features, dim=1)
            combined_feature, att = self.attention(features)
        else:
            print("Please using a correct fusion type")
            exit()
        return combined_feature
class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, temperature, bias):
        super(Discriminator, self).__init__()
        self.aligner = (nn.Linear(in_dim, hidden_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, 1))
        self.temperature = temperature
        self.bias = bias
    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2
    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (
                1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(edges_weights_raw.device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        output = torch.sigmoid(gate_inputs).squeeze()
        return output
    def forward(self, embedding, edges):
        embedding_ = self.aligner(embedding)
        edges_weights_raw = self.get_edge_weight(embedding_, edges)
        weights = self.gumbel_sampling(edges_weights_raw)
        return weights

def cal_similarity_graph(node_embeddings, right_node_embedding=None):
    if right_node_embedding is None:
        similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    else:
        similarity_graph = torch.mm(node_embeddings, right_node_embedding.t())
    return similarity_graph
class InnerProductSimilarity:
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()
    def __call__(self, embeddings, right_embeddings=None):
        similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities
class CosineSimilarity:
    def __init__(self):
        super(CosineSimilarity, self).__init__()
    def __call__(self, embeddings, right_embeddings=None):
        if right_embeddings is None:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            right_embeddings = F.normalize(right_embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities
class WeightedCosine(nn.Module):
    def __init__(self, input_size, num_pers):
        super().__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
    def __call__(self, embeddings):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(embeddings.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
        embeddings_fc = embeddings.unsqueeze(0) * expand_weight_tensor
        embeddings_norm = F.normalize(embeddings_fc, p=2, dim=-1)
        attention = torch.matmul(
            embeddings_norm, embeddings_norm.transpose(-1, -2)
        ).mean(0)
        return attention
class MLPRefineSimilarity(nn.Module):
    def __init__(self, hid, dropout):
        super(MLPRefineSimilarity, self).__init__()
        self.gen_mlp = nn.Linear(2 * hid, 1)
        self.dropout = nn.Dropout(dropout)
    def __call__(self, embeddings, v_indices):
        num_node = embeddings.shape[0]
        f1 = embeddings[v_indices[0]]
        f2 = embeddings[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node)).to_dense()
        return z_matrix
# class Minkowski:

class BaseLearner(nn.Module):
    """Abstract base class for graph learner"""
    def __init__(self, metric, processors):
        super(BaseLearner, self).__init__()

        self.metric = metric
        self.processors = processors

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MLPLearner(BaseLearner):
    """Multi-Layer Perception learner"""
    def __init__(self, metric, processors, nlayers, in_dim, hidden_dim, activation, sparse, k=None):
        super().__init__(metric=metric, processors=processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, in_dim))
        for _ in range(1, nlayers):
            self.layers.append(nn.Linear(in_dim, in_dim))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.param_init()
        self.activation = activation
        self.sparse = sparse
        self.k = k
    def param_init(self):
        for layer in self.layers:
            # layer.reset_parameters()
            layer.weight = nn.Parameter(torch.eye(self.in_dim))
    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x
    def forward(self, features):
        if self.sparse:
            z = self.internal_forward(features)
            # z=features
            # TODO: knn should be moved to processor
            rows, cols, values = knn_fast(z, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = self.activation(values_)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device=features.device)
            adj.edata['w'] = values_
            return adj
        else:
            z = self.internal_forward(features)
            # z=features
            z = F.normalize(z, dim=1, p=2)
            similarities = self.metric(z)
            for processor in self.processors:
                similarities = processor(similarities)
            similarities = F.relu(similarities)
            return similarities
class SpLearner(BaseLearner):
    """Sparsification learner"""
    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__(metric=metric, processors=processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim * 2 + 1, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, 1))
        self.param_init()
        self.activation = activation
        self.k = k
        self.weight = weight
    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x
    def gumbel_softmax_sample(self, logits, temperature, adj, training):
        """Draw a sample from the Gumbel-Softmax distribution"""
        r = self.sample_gumble(logits.coalesce().values().shape)
        if training is not None:
            values = torch.log(logits.coalesce().values()) + r.to(logits.device)
        else:
            values = torch.log(logits.coalesce().values())
        values /= temperature
        y = torch.sparse_coo_tensor(indices=adj.coalesce().indices(), values=values, size=adj.shape)
        return torch.sparse.softmax(y, dim=1)
    def sample_gumble(self, shape):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)
    def forward(self, features, adj, temperature, training=None):
        indices = adj.coalesce().indices().t()
        values = adj.coalesce().values()
        f1_features = torch.index_select(features, 0, indices[:, 0])
        f2_features = torch.index_select(features, 0, indices[:, 1])
        auv = torch.unsqueeze(values, -1)
        temp = torch.cat([f1_features, f2_features, auv], -1)
        temp = self.internal_forward(temp)
        z = torch.reshape(temp, [-1])
        z_matrix = torch.sparse_coo_tensor(indices=indices.t(), values=z, size=adj.shape)
        pi = torch.sparse.softmax(z_matrix, dim=1)
        y = self.gumbel_softmax_sample(pi, temperature, adj, training)
        y_dense = y.to_dense()
        top_k_v, top_k_i = torch.topk(y_dense, self.k)
        kth = torch.min(top_k_v, -1)[0] + eps
        kth = kth.unsqueeze(-1)
        kth = torch.tile(kth, [1, kth.shape[0]])
        mask2 = y_dense >= kth
        mask2 = mask2.to(torch.float32)
        row_sum = mask2.sum(-1)
        dense_support = mask2
        if self.weight:
            dense_support = torch.mul(y_dense, mask2)
        else:
            print('no gradient bug here!')
            exit()
        # norm
        dense_support = dense_support + torch.eye(adj.shape[0]).to(dense_support.device)
        rowsum = torch.sum(dense_support, -1) + 1e-6
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_mat_inv_sqer = torch.diag(d_inv_sqrt)
        ad = torch.matmul(dense_support, d_mat_inv_sqer)
        adt = ad.t()
        dadt = torch.matmul(adt, d_mat_inv_sqer)
        support = dadt
        return support
