import numpy as np
import torch
# from autoattack import AutoAttack

#@title deepfool attack
def deepfool(model, x, max_iter=50):
    """DeepFool attack"""
    nx = x.clone()
    nx.requires_grad_()
    eta = torch.zeros(nx.shape).cuda() if torch.cuda.is_available() else torch.zeros(nx.shape)

    out = model(nx+eta)
    n_class = out.shape[1]
    initial_label = out.max(1)[1]
    pred_nx = out.max(1)[1]

    i_iter = 0
    cumul_dis_2 = 0.
    cumul_dis_inf = 0.
    while pred_nx == initial_label and i_iter < max_iter:
        out[0, pred_nx].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        w_l = None
        for i in range(n_class):
            if i == initial_label:
                continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, initial_label]
            value_i = np.abs(fi.item()) / torch.norm(wi.flatten())
            # breakpoint()
            if value_i < value_l:
                value_l = value_i
                w_l = wi

        ri = value_l/torch.norm(w_l.flatten()) * w_l
        eta += ri.clone()
        nx.grad.data.zero_()
        out = model(nx+eta)
        pred_nx = out.max(1)[1]
        i_iter += 1
    return x+ri

def is_attack_successful(model, x, x_adv):
    out = model(x)
    out_adv = model(x_adv)
    _, y = torch.max(out.data, 1)
    _, y_adv = torch.max(out_adv.data, 1)
    return y != y_adv

