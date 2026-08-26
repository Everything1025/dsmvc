import numpy as np
import torch
from munkres import Munkres
from sklearn import metrics

def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=====Model Parameters====')
    print('Total:{:.3f}M,\nTrainable: {:.3f}M'.format(total_num / 1e6, trainable_num / 1e6))
def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = np.matmul(association_mat, sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')

def evaluate(y_pre_sp, association_mat, gt):
    if isinstance(y_pre_sp, torch.Tensor):
        y_pre_sp = y_pre_sp.cpu().numpy()
    if isinstance(association_mat, torch.Tensor):
        association_mat = association_mat.cpu().numpy()
    y_pre_pixel = spixel_to_pixel_labels(y_pre_sp, association_mat)
    class_map = y_pre_pixel.reshape(gt.shape)
    indx = np.where(gt != 0)
    y_target = gt[indx]
    y_predict = class_map[indx]
    y_predict, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_target, y_predict, return_aligned=True)
    return (acc, kappa, nmi, ari, pur), ca, (indx, y_target.astype(int), y_predict)
def print_table(data, headers, col_width=12):
    num_cols = len(headers)
    col_spacing = 3
    row_format = "".join(["{:>" + str(col_width) + "}"] * (num_cols))
    row_separator = "".join(["{:->" + str(col_width) + "}"] * (num_cols))
    print(row_format.format(*headers))
    print(row_separator.format(*["" for _ in range(num_cols)]))
    for i, row in enumerate(data):
        row_values = [f"{val:.4f}" if isinstance(val, float) else val for val in row]
        print(row_format.format(*row_values))

