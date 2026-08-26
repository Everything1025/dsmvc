import argparse
import os
import random
from copy import deepcopy
from time import time
from warnings import simplefilter

import numpy as np
import torch
from sklearn.cluster import KMeans

from config import DEFAULT_CONFIG
from losses import loss_multiview2
from models.network import MultiViewModel
from utils.graph_utils import delete_graph_all
from utils.metrics import evaluate, print_table

try:
    from torch_clustering import FaissKMeans
except Exception:
    FaissKMeans = None
try:
    from torch_clustering import PyTorchKMeans
except Exception:
    PyTorchKMeans = None

simplefilter(action='ignore', category=FutureWarning)

def rand_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def train(pretrain=True):
    # rand_seed(args.seed)
    dataset_name = args.dataset_name  # 仍需从命令行获取

    # 仅使用预处理的固定文件名（位于 save/）
    pt_path = f"save/{dataset_name}_graphs.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f'Preprocessed .pt file not found: {pt_path}')

    data = torch.load(pt_path)
    x_list = [t.float() for t in data['x']]
    adj_list = [a.float() for a in data['adj']]
    y_full = data['y'].long()
    assoc_full = data.get('association_mat', None)
    n_clusters = int(data.get('n_classes', (y_full.unique().numel() - 1)))

    # 要求 .pt 中必须包含 ground-truth 和 association_mat
    gt = data.get('gt', None)
    if gt is None:
        raise RuntimeError(f'GT not found inside {pt_path}. Recreate .pt including gt.')
    if assoc_full is None:
        raise RuntimeError(f'association_mat not found inside {pt_path}. Recreate .pt including association_mat.')

    # 处理是否去除背景：训练时使用缩减后的超像素集合，但保留完整 association_mat 供评估使用
    if DEFAULT_CONFIG.get('remove_bkg', True):
        non_zero_idx = (y_full != 0).nonzero(as_tuple=True)[0]
        x_train = [t[non_zero_idx] for t in x_list]
        adjs_train = [a[non_zero_idx][:, non_zero_idx] for a in adj_list]
    else:
        non_zero_idx = None
        x_train = x_list
        adjs_train = adj_list

    # 构造一个轻量 dataset 对象以兼容后续调用（association_mat 保持完整）
    class LoadedDataset:
        def __init__(self, x_full, x_train, adjs_train, y_full, assoc_full, gt, n_clusters, non_zero_idx):
            self.x_full = x_full
            self.x = x_train
            self.adj = adjs_train
            self.y = y_full
            self.association_mat = assoc_full
            self.gt = gt
            self.n_classes = n_clusters
            self.non_zero_idx = non_zero_idx

        def recover_background(self, sp_pred):
            import numpy as _np
            if isinstance(sp_pred, torch.Tensor):
                sp_pred = sp_pred.detach().cpu().numpy()
            full = _np.zeros(self.y.shape[0], dtype=sp_pred.dtype)
            if self.non_zero_idx is not None:
                full[self.non_zero_idx.cpu().numpy()] = sp_pred
            else:
                full[:] = sp_pred
            return full

        def __len__(self):
            return int(self.x[0].shape[0])

    dataset = LoadedDataset(x_list, x_train, adjs_train, y_full, assoc_full, gt, n_clusters, non_zero_idx)

    # 读取配置信息（直接从 DEFAULT_CONFIG）
    cfg = DEFAULT_CONFIG.get(dataset_name, {})
    n_samples = len(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = int(DEFAULT_CONFIG.get('n_epoches', 50))
    save_path = os.path.join(DEFAULT_CONFIG.get('save_path', './save'), f"{dataset_name}.pkl")

    # 将 tensors 移动到 device，确保形状为 (n_samples, -1)
    x = [t.reshape(n_samples, -1).float().to(device) for t in dataset.x]
    adjs = [a.float().to(device) for a in dataset.adj]
    # 从配置读取模型超参数
    nlayers = int(cfg.get('nlayers', 4))
    dropout = float(cfg.get('dropout', 0.5))
    in_dim = int(cfg.get('in_dim', 128))
    hidden_dim = int(cfg.get('hidden_dim', 256))
    emb_dim = int(cfg.get('emb_dim', 512))
    proj_dim = int(cfg.get('proj_dim', 512))
    temperature = float(cfg.get('temperature', 1.0))
    bias = float(cfg.get('bias', 1e-05))
    sparse_flag = bool(DEFAULT_CONFIG.get('sparse', False))

    model = MultiViewModel(
        nlayers=nlayers,
        in_channels=[it.shape[1] for it in x],  # 动态获取输入维度
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        sparse=sparse_flag,
        temperature=temperature,
        bias=bias,
        num_g=len(adjs),
        n_clusters=n_clusters,
        device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('lr', 1e-5)))
    if torch.cuda.is_available():
        model = model.cuda()
        x = [f.cuda() for f in x]
        adjs = [a.cuda() for a in adjs]
    # Prefer PyTorchKMeans (if available), then GPU FaissKMeans, else sklearn KMeans
    if PyTorchKMeans is not None and torch.cuda.is_available():
        try:
            kmeans = PyTorchKMeans(n_clusters=n_clusters)
        except Exception:
            # fallback to simple constructor
            kmeans = PyTorchKMeans(n_clusters=n_clusters)
    elif FaissKMeans is not None and torch.cuda.is_available():
        kmeans = FaissKMeans(metric='euclidean', n_clusters=n_clusters, n_init=10, max_iter=300,
                             random_state=1234, distributed=False, verbose=False)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)

    def kmeans_predict(km, tensor_z):
        # tensor_z: torch.Tensor on device
        # handle PyTorchKMeans and FaissKMeans which may accept torch tensors
        if PyTorchKMeans is not None and isinstance(km, PyTorchKMeans):
            labels = km.fit_predict(tensor_z)
            if isinstance(labels, torch.Tensor):
                return labels.cpu().numpy()
            return labels
        if FaissKMeans is not None and isinstance(km, FaissKMeans):
            labels = km.fit_predict(tensor_z)
            if isinstance(labels, torch.Tensor):
                return labels.cpu().numpy()
            else:
                return labels
        else:
            return km.fit_predict(tensor_z.cpu().numpy())

    t0 = time()
    if pretrain:
        print('start training.')
        best_acc = [0] * 5
        for epoch in range(1, EPOCH):
            model.train()
            print(f'[epoch {epoch}] forward start')
            specific_adjs, aug_adjs, z_specific_adjs, z_combined, z_proj_s, z_proj_u, z_proj_a = model(
                x, adjs)
            print(f'[epoch {epoch}] forward done')
            if (epoch % 5 == 0) and bool(DEFAULT_CONFIG.get('refine', True)):
                z = z_combined.detach()
                print(f'[epoch {epoch}] kmeans (refine) start')
                y_pred = kmeans_predict(kmeans, z)
                print(f'[epoch {epoch}] kmeans (refine) done')
                adjs = delete_graph_all(labels=torch.from_numpy(y_pred).to(device), adjs=adjs)
            loss = loss_multiview2(specific_adjs, aug_adjs, z_specific_adjs, z_proj_s, z_proj_u, z_proj_a, adjs,
                                   w_smi=float(cfg.get('w_smi', 1)),
                                   w_umi=float(cfg.get('w_umi', 1)),
                                   w_dis=float(cfg.get('w_dis', 0.1)),
                                   top_k_rate=float(cfg.get('top_k_rate', 0.001)), temperature=float(cfg.get('temperature', 1.0)))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f'epoch:{epoch}, loss:{loss.item()}')
            model.eval()
            with torch.no_grad():
                print(f'[epoch {epoch}] eval forward start')
                specific_adjs, aug_adjs, z_specific_adjs, z_combined, z_proj_s, z_proj_u, z_proj_a = model(
                    x, adjs)
                print(f'[epoch {epoch}] eval forward done')
            z = z_combined.detach()
            print(f'[epoch {epoch}] kmeans start')
            y_pred = kmeans_predict(kmeans, z)
            print(f'[epoch {epoch}] kmeans done')
            if DEFAULT_CONFIG.get('remove_bkg', True):
                y_pred_all = dataset.recover_background(y_pred)
            print(f'[epoch {epoch}] evaluate start')
            mts0, ca, eta = evaluate(y_pred_all, dataset.association_mat, dataset.gt)
            print(f'[epoch {epoch}] evaluate done')
            if mts0[0] > best_acc[0]:
                best_acc = mts0
                weights = deepcopy(model.state_dict())
            print_table([mts0], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
        torch.save(weights, save_path)
    return best_acc, time() - t0

def main():
    global args
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_name', type=str,default='Trento', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    rand_seed(args.seed)

    # 使用默认配置并设置 GPU（直接使用 DEFAULT_CONFIG）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEFAULT_CONFIG.get('gpu_id', '0'))
    os.makedirs(DEFAULT_CONFIG.get('save_path', './save'), exist_ok=True)

    best_acc, t = train(pretrain=True)
    print()
    print_table([best_acc], ['ACC', 'Kappa', 'NMI', 'ARI', 'Purity'])
    print(f'total time cost:{t}s')


if __name__ == "__main__":
    main()
