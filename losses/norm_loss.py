import torch
import torch.nn as nn
import torch.nn.functional as F


log_sigmoid = torch.nn.LogSigmoid()


def calculate_norm_loss(features, mask):
    """
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    Aa = A[mask == 1]
    if torch.sum(mask == 1) != 0:  # second stage, and exist anomalies
        r_max = min(0.9 * Aa.min().item(), 0.4)  # get the minimum abnormal radii as r_max
        r_min = r_max * 0.99  
    else:  # first stage, or no anomalies in second stage
        r_max = 0.4
        r_min = 0.99 * 0.4
    
    loss, loss_n, loss_a = 0, 0, 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        An_larger = An[An > r_max]  # larger than r_max
        An_lower = An[An < r_min]  # lower than r_min
        if An_larger.shape[0] != 0:
            weights = torch.exp(An_larger - r_max).detach()
            loss_larger = torch.mean(-log_sigmoid(-(An_larger - r_max)) * weights)
        else:
            loss_larger = 0
        if An_lower.shape[0] != 0:
            weights = torch.exp(r_min - An_lower).detach()
            loss_lower = torch.mean(-log_sigmoid(-(r_min - An_lower)) * weights)
        else:
            loss_lower = 0
        
        loss_n = loss_larger + loss_lower
        loss += loss_n

    if torch.sum(mask == 1) != 0:
        boundary = r_max + 0.75
        # using log barrier loss to push ano features out the boundary
        lower_indicator = Aa < boundary
        Aa_lower = Aa[lower_indicator]  
        if Aa_lower.shape[0] != 0:  # for norm < boundary, pushing these features out
            weights = torch.exp(boundary - Aa_lower).detach()
            loss_lower = torch.mean(-log_sigmoid(-(boundary - Aa_lower)) * weights)
        else:
            loss_lower = 0
            
        loss_a = loss_lower
        loss += loss_a

    return loss