import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_angle_loss(features1, features2, masks, center: torch.Tensor = None, temp: float = 0.15, reduction: str = 'mean'):
    if center is not None:
        features1 = features1 - center
        features2 = features2 - center
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)
    N = features1.size(0)

    features = torch.cat([features1, features2], dim=0)  # (2N, D)    
    masks_ = torch.cat([masks, masks], dim=0)  # (2N, )
    nor_features = features[masks_ == 0]  # (N1, )
    ano_features = features[masks_ == 1]  # (N2, )
    
    # as number of ano features is low, this will cause the contrastive_losses_part1 be problematical
    # Thus, for normal features, we also use all normal and abnormal features as negative part
    # TODO: another thing can try, lower down the weights when normal is as negative
    # nor_neg_similarities = torch.exp(torch.mm(nor_features, features.t().contiguous()) / temp)  # (N1, 2N)
    nor_neg_similarities = torch.exp(torch.mm(nor_features, ano_features.t().contiguous()) / temp)  # (N1, N2)
    ano_neg_similarities = torch.exp(torch.mm(ano_features, nor_features.t().contiguous()) / temp)  # (N2, N1)

    pos_similarities = torch.exp(torch.sum(features1 * features2, dim=-1) / temp)  # (N, ), the similarities between each feature and its aug view
    pos_similarities = torch.cat([pos_similarities, pos_similarities], dim=0)   # (2N, )
    nor_pos_similarities = pos_similarities[masks_ == 0]  # (N1, )
    ano_pos_similarities = pos_similarities[masks_ == 1]  # (N2, )
    
    contrastive_losses_part1 = -torch.log(nor_pos_similarities / nor_neg_similarities.sum(dim=-1))  # as number of ano features is low, the loss may be negative
    contrastive_losses_part2 = -torch.log(ano_pos_similarities / ano_neg_similarities.sum(dim=-1))
    contrastive_losses = torch.cat([contrastive_losses_part1, contrastive_losses_part2], dim=0)
    
    if reduction == 'mean':
        loss = torch.mean(contrastive_losses)
    elif reduction == 'sum':
        loss = torch.sum(contrastive_losses)
    else:
        raise RuntimeError(f"The loss reduction '{reduction}' is not supported!")
        
    return loss