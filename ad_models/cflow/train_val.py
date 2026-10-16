import time
import random
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model import load_decoder_arch, positionalencoding2d
from .utils import *
from .metrics import calculate_metrics


gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def train_meta_epoch(c, epoch, loader, encoder, projectors, decoders, ref_features, optimizer, with_pretrained, N, class_name):
    P = c['condition_vec']
    L = c['pool_layers']
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c['sub_epochs']):
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c['sub_epochs'], optimizer)
            # sample batch
            try:
                image, _, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _, _ = next(iterator)
            # encoder prediction
            image = image.to(c['device'])  # single scale
            with torch.no_grad():
                features = encoder.encode_image_from_tensors(image)
                
                if with_pretrained:
                    # random reference samples
                    # ref_features = get_reference_features(encoder, [class_name] * b, c['device'])
                    # pfeatures = get_mc_matched_reference_features(features, [class_name] * b, ref_features)
                    # fixed reference samples
                    pfeatures = get_matched_ref_features(features, ref_features)
                    rfeatures = get_residual_features(features, pfeatures)
                    rfeatures = projectors(*rfeatures)
                    features = rfeatures[-L:]
                else:
                    features = features[-L:]
            
            # train decoder
            for l in range(L):
                e = features[l]
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(c['device']).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(c['device'])  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c['dec_arch']:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        if c['verbose']:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def test_meta_epoch(c, epoch, loader, encoder, projectors, decoders, ref_features, with_pretrained, N):
    # test
    if c['verbose']:
        print('\nCompute loss and scores on test set:')
    #
    P = c['condition_vec']
    L = c['pool_layers']
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for _ in range(L)]
    test_loss = 0.0
    test_count = 0
    start = time.time()
    with torch.no_grad():
        for idx, (image, label, mask, _) in enumerate(tqdm(loader, disable=c['hide_tqdm_bar'])):
            # save
            gt_label_list.append(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze(1).cpu().numpy())
            # data
            image = image.to(c['device']) # single scale
            
            features = encoder.encode_image_from_tensors(image)
            
            if with_pretrained:
                pfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, pfeatures)
                rfeatures = projectors(*rfeatures)
                features = rfeatures[-L:]
            else:
                features = features[-L:]
            
            # test decoder
            for l in range(L):
                e = features[l]
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if idx == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(c['device']).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                decoder = decoders[l]
                #
                if 'cflow' in c['dec_arch']:
                    z, log_jac_det = decoder(e_r, [c_r,])
                else:
                    z, log_jac_det = decoder(e_r)
                #
                decoder_log_prob = get_logp(C, z, log_jac_det)
                log_prob = decoder_log_prob / C  # likelihood per dim
                loss = -log_theta(log_prob)
                test_loss += t2np(loss.sum())
                test_count += len(loss)
                test_dist[l].append(log_prob.reshape(B, H, W).cpu())
    #
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c['verbose']:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    #
    return height, width, test_dist, gt_label_list, gt_mask_list


def train_val(c, encoder, projectors, train_loader, test_loader, ref_features, class_name, with_pretrained):
    init_seeds(seed=0)
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    L = c['pool_layers'] # number of pooled layers
    print('Number of pool layers =', L)
    pool_dims = encoder.feature_dimensions
    pool_dims = pool_dims[-L:]  # last L layers
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c['device']) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c['lr'])
   
    N = 4096  # hyperparameter that increases batch size for the decoder model by N
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))
    if c['action_type'] == 'norm-test':
        c['meta_epochs'] = 1
    
    img_aucs, img_aps, img_f1_scores, pix_aucs, pix_aps, pix_f1_scores, pix_aupros = [], [], [], [], [], [], []
    for epoch in range(c['meta_epochs']):
        if c['action_type'] == 'norm-test' and c['checkpoint']:
            load_weights(encoder, decoders, c['checkpoint'])
        elif c['action_type'] == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, encoder, projectors, decoders, ref_features, optimizer, with_pretrained, N, class_name)
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c['action_type']))
        
        height, width, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
            c, epoch, test_loader, encoder, projectors, decoders, ref_features, with_pretrained, N)

        # PxEHW
        print('Heights/Widths', height, width)
        test_map = [list() for _ in range(L)]
        for l in range(L):
            test_norm = torch.cat(test_dist[l], dim=0).to(torch.double)
            # test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=224, mode='bilinear', align_corners=True).squeeze().numpy()
        # score aggregation
        score_map = np.zeros_like(test_map[0])
        for l in range(L):
            score_map += test_map[l]
        score_mask = score_map
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
        # calculate detection AUROC
        score_label = np.max(super_mask, axis=(1, 2))
        gt_label = np.concatenate(gt_label_list, axis=0)
        gt_mask = np.concatenate(gt_mask_list, axis=0)
        
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(score_label, super_mask, gt_label, gt_mask, pro=True, only_max_value=True)
        print("Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
        img_aucs.append(img_auc)
        img_aps.append(img_ap)
        img_f1_scores.append(img_f1_score)
        pix_aucs.append(pix_auc)
        pix_aps.append(pix_ap)
        pix_f1_scores.append(pix_f1_score)
        pix_aupros.append(pix_aupro)
    img_auc = np.max(img_aucs)
    img_ap = np.max(img_aps)
    img_f1_score = np.max(img_f1_scores)
    pix_auc = np.max(pix_aucs)
    pix_ap = np.max(pix_aps)
    pix_f1_score = np.max(pix_f1_scores)
    pix_aupro = np.max(pix_aupros)
    print("Class Name: {}, Best Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro