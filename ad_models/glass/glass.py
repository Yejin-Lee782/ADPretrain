
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm

import cv2
import glob
import shutil
from .metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro, calculate_metrics
from .utils import torch_format_2_numpy_img, del_remake_dir, distribution_judge, get_matched_ref_features, get_residual_features
from .common import Aggregator, Preprocessing, NetworkFeatureAggregator, RescaleSegmentor
from .loss import FocalLoss
from .model import Discriminator, Projection, PatchMaker

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            projectors,
            device,
            input_shape,
            with_pretrained,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.projectors = projectors
        self.input_shape = input_shape
        self.device = device
        self.with_pretrained = with_pretrained

        self.forward_modules = torch.nn.ModuleDict({})
        
        feature_dimensions = self.backbone.feature_dimensions
        feature_dimensions = feature_dimensions[-2:]  # only last two layers
        self.forward_modules["feature_aggregator"] = self.backbone

        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, ref_features, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"].encode_image_from_tensors(images)
                
                if self.with_pretrained:
                    mfeatures = get_matched_ref_features(features, ref_features)
                    rfeatures = get_residual_features(features, mfeatures)
                    if self.projectors is not None:
                        rfeatures = self.projectors(*rfeatures)
                else:
                    if self.projectors is not None:
                        features = self.projectors(*features)

        if self.with_pretrained:
            features = rfeatures[-2:]  # only last two layers are used
        else:
            features = features[-2:]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def trainer(self, training_data, val_data, ref_features, name):
        state_dict = {}
        # ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        # ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        # if len(ckpt_path) != 0:
        #     LOGGER.info("Start testing, ckpt file found!")
        #     return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        self.distribution = training_data.dataset.distribution
        xlsx_path = './ad_models/glass/datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = distribution_judge(avg_img, name)
            os.makedirs(f'./ad_models/glass/results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./ad_models/glass/results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_img_auc, best_img_ap, best_pix_auc, best_pix_ap, best_pix_aupro = 0, 0, 0, 0, 0
        for i_epoch in pbar:
            self.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, ref_features, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = self._embed(img, ref_features, evaluation=False)[0]
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])  # (B, L, C)

                    batch_mean = torch.mean(outputs, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)  # normal feature centers, (L, C)

            pbar_str, pt, pf = self._train_discriminator(training_data, ref_features, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, img_scores, pix_scores, labels_gt, masks_gt = self.predict(val_data, ref_features)
                img_auc, img_ap, pix_auc, pix_ap, pix_aupro = self._evaluate(images, img_scores, pix_scores,
                                                                             labels_gt, masks_gt, name, path='eval')

                print()
                print("Class Name: {}, Image AUC | AP: {} | {}, Pixel AUC | AP | AUPRO: {} | {} | {}".format(
                    name, img_auc, img_ap, pix_auc, pix_ap, pix_aupro))
            
                if img_auc > best_img_auc:
                    best_img_auc = img_auc
                if img_ap > best_img_ap:
                    best_img_ap = img_ap
                if pix_auc > best_pix_auc:
                    best_pix_auc = pix_auc
                if pix_ap > best_pix_ap:
                    best_pix_ap = pix_ap
                if pix_aupro > best_pix_aupro:
                    best_pix_aupro = pix_aupro

                pbar.set_description_str(pbar_str)

            # torch.save(state_dict, ckpt_path_save)
        
        return best_img_auc, best_img_ap, best_pix_auc, best_pix_ap, best_pix_aupro

    def _train_discriminator(self, input_data, ref_features, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)  # pseudo anomaly images
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, ref_features, evaluation=False)[0])
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                true_feats = self.pre_projection(self._embed(img, ref_features, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
            else:
                fake_feats = self._embed(aug, ref_features, evaluation=False)[0]
                fake_feats.requires_grad = False
                true_feats = self._embed(img, ref_features, evaluation=False)[0]
                true_feats.requires_grad = True

            mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device)
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)  # (B*L, C)
            gaus_feats = true_feats + noise  # anomaly features

            center = self.c.repeat(img.shape[0], 1, 1)  # (B, L, C)
            center = center.reshape(-1, center.shape[-1])
            true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)  # normal features
            c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

            for step in range(self.step + 1):  # global anomaly generating
                scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
                true_scores = scores[:len(true_feats)]
                gaus_scores = scores[len(true_feats):]
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]  # (B*L, C)
                grad_norm = torch.norm(grad, dim=1)
                grad_norm = grad_norm.view(-1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    proj_feats = center if self.svd == 1 else true_feats
                    r = r_t if self.svd == 1 else 0.5

                    h = gaus_feats - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h = proj * h
                    gaus_feats = proj_feats + h

            fake_points = fake_feats[mask_s_gt[:, 0] == 1]  # anomaly features
            true_points = true_feats[mask_s_gt[:, 0] == 1]  # normal features in anomaly position
            c_f_points = center[mask_s_gt[:, 0] == 1]
            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            proj_feats = c_f_points if self.svd == 1 else true_points
            r = r_t if self.svd == 1 else 1

            if self.svd == 1:
                h = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h = proj * h
                fake_points = proj_feats + h
                fake_feats[mask_s_gt[:, 0] == 1] = fake_points

            fake_scores = self.discriminator(fake_feats)
            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2
                d_hard = torch.quantile(fake_dist, q=self.p)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, ref_features, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data, ref_features)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        img_min_scores = min(scores)
        img_max_scores = max(scores)
        norm_scores = (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-10)

        image_scores = compute_imagewise_retrieval_metrics(norm_scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = np.min(segmentations)
            max_scores = np.max(segmentations)
            norm_segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-10)

            pixel_scores = compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = compute_pro(np.squeeze(np.array(masks_gt)), norm_segmentations)

                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.

        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        # defects = np.array(images)
        # targets = np.array(masks_gt)
        # for i in range(len(defects)):
        #     defect = torch_format_2_numpy_img(defects[i])
        #     target = torch_format_2_numpy_img(targets[i])

        #     mask = cv2.cvtColor(cv2.resize(norm_segmentations[i], (defect.shape[1], defect.shape[0])),
        #                         cv2.COLOR_GRAY2BGR)
        #     mask = (mask * 255).astype('uint8')
        #     mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        #     img_up = np.hstack([defect, target, mask])
        #     img_up = cv2.resize(img_up, (256 * 3, 256))
        #     full_path = './results/' + path + '/' + name + '/'
        #     del_remake_dir(full_path, del_flag=False)
        #     cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader, ref_features):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    # images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image, ref_features)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, img, ref_features):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes = self._embed(img, ref_features, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
