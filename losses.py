import torch
import torch.nn.functional as F

def calc_lower_bound(x, x_aug, pos, temperature=0.2, sym=True):
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-8)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix.mul(pos)
    if sym:
        loss_0 = pos_sim.sum(dim=0) / (sim_matrix.sum(dim=0))
        loss_1 = pos_sim.sum(dim=1) / (sim_matrix.sum(dim=1))
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    return loss
def sce_s_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = ((x * y).sum(dim=-1)).pow_(beta)
    loss = loss.mean()
    return loss
def loss_multiview2(specific_adjs, aug_adjs, z_specific_adjs, z_proj_s,
                    z_proj_u, z_proj_a, adjs, w_smi, w_umi, w_dis, top_k_rate, temperature):
    if w_smi != 0:
        loss_smi = 0
        cnt = 0
        adjs_all = torch.zeros_like(adjs[0])
        for adj, adj_aug in zip(specific_adjs, aug_adjs):
            adjs_all += adj + adj_aug
        batch_size, _ = adjs_all.size()
        pos_eye = torch.eye(batch_size).to(z_specific_adjs[0].device)
        topk_k = int(batch_size * top_k_rate)
        for i in range(batch_size):
            row = adjs_all[i]
            row[i] = 0
            topk_values, topk_indices = torch.topk(row, k=topk_k)
            pos_eye[i, topk_indices] = 1
        for i in range(len(adjs)):
            for j in range(i + 1, len(adjs)):
                loss_smi += calc_lower_bound(z_proj_a[i], z_proj_a[j], pos_eye, temperature)
                cnt += 1
        loss_smi = loss_smi / cnt
    if w_umi != 0:
        loss_umi = 0
        for i in range(len(adjs)):
            adjs_all = specific_adjs[i] + aug_adjs[i]
            pos_eye = torch.eye(batch_size).to(z_specific_adjs[0].device)
            topk_k = int(batch_size * top_k_rate)
            for j in range(batch_size):
                row = adjs_all[j]
                row[j] = 0
                topk_values, topk_indices = torch.topk(row, k=topk_k)
                pos_eye[j, topk_indices] = 1
            loss_umi += calc_lower_bound(z_proj_s[i], z_proj_u[i], pos_eye, temperature)
    if w_dis != 0:
        loss_dis = 0
        for i in range(len(adjs)):
            if specific_adjs[i].is_sparse:
                adj_dense = specific_adjs[i].to_dense()
            else:
                adj_dense = specific_adjs[i]
            if aug_adjs[i].is_sparse:
                adj_aug_dense = aug_adjs[i].to_dense()
            else:
                adj_aug_dense = aug_adjs[i]
            loss_dis += sce_s_loss(adj_dense, adj_aug_dense)
    loss = w_smi * loss_smi + w_umi * loss_umi + w_dis * loss_dis
    return loss
