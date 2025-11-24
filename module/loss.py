
import torch
import torch.nn.functional as F

def entropy(pred, target, temp_inv, use_norm=False):

    if use_norm:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

    logits = temp_inv * pred @ target.T

    p = F.softmax(logits, dim=-1)

    loss = - torch.sum(p * torch.log(p), dim=1).mean()

    return loss


def js_div(y_pred, y):
    y_pred_prob = F.softmax(y_pred, dim=1)
    y_prob = F.softmax(y, dim=1)
    m_prob = (y_pred_prob + y_prob) / 2
    m_log = m_prob.log()

    return F.kl_div(m_log, y_pred_prob, reduction="batchmean", log_target=False) + \
        F.kl_div(m_log, y_prob, reduction="batchmean", log_target=False)