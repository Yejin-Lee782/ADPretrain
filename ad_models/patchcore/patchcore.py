"""PatchCore and PatchCore detection methods."""
import logging
import os
import json
import timm
import pickle
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torchvision.transforms as T

from .common import FaissNN, Preprocessing, Aggregator, NearestNeighbourScorer, RescaleSegmentor
from .sampler import IdentitySampler
from .utils import get_matched_ref_features, get_residual_features

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device, dtype=torch.float):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.dtype = dtype

    def load(
        self,
        encoder,
        projectors,
        layers_to_extract_from,
        device,
        residual,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=IdentitySampler(),
        nn_method=FaissNN(False, 4),
        **kwargs,
    ):
        self.encoder = encoder
        self.projectors = projectors
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.residual = residual
        
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_dimensions = encoder.feature_dimensions
        self.encoder_name = encoder.name
        
        feature_dimensions = feature_dimensions[-3:]  # last three layers
        preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, ref_features, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features
        
        with torch.no_grad():
            features = self.encoder.encode_image_from_tensors(images.to(dtype=self.dtype), shape='img')
            
            if self.residual:
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures)
                if self.projectors is not None:
                    rfeatures = self.projectors(*rfeatures)
            else:
                if self.projectors is not None:
                    features = self.projectors(*features)

        if self.residual:
            features = [x.to(torch.float) for x in rfeatures]
        else:
            features = [x.to(torch.float) for x in features]
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features[-3:]
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
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
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]  # (B*H*W, C, 3, 3), (3, 3) is the local patch size

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data, ref_features, class_name, aligned=False):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, ref_features, class_name, aligned)

    def _fill_memory_bank(self, input_data, ref_features, class_name, aligned=False):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image, ref_features):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image, ref_features)

        all_features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for idx, data_item in enumerate(data_iterator):
                image, _, _, _ = data_item
                    
                all_features.append(_image_to_features(image, ref_features))

        # features = np.concatenate(all_features, axis=0)
        # # print(features.shape)
        # features = self.featuresampler.run(features)
        # # print(features.shape)

        # self.anomaly_scorer.fit(detection_features=[features])
        features = np.concatenate(
            all_features,
            axis=0
        )

        print(
            "[PatchCore] Train features before sampling:",
            features.shape
        )

        # 원본 normal feature 보관
        self.train_features_full = features.copy()

        features = self.featuresampler.run(
            features
        )

        print(
            "[PatchCore] Memory bank features:",
            features.shape
        )

        # 실제 PatchCore가 사용하는 memory bank feature
        self.memory_bank_features = features.copy()

        self.anomaly_scorer.fit(
            detection_features=[features]
        )
    def predict(
        self,
        data,
        ref_features,
        class_name,
        aligned=False,
        return_features=False,
    ):
        if isinstance(data, torch.utils.data.DataLoader):

            return self._predict_dataloader(
                data,
                ref_features,
                class_name,
                aligned,
                return_features=return_features,
            )

        return self._predict(
            data,
            ref_features,
            return_features=return_features,
        )
    # def predict(self, data, ref_features, class_name, aligned=False):
    #     if isinstance(data, torch.utils.data.DataLoader):
    #         return self._predict_dataloader(data, ref_features, class_name, aligned)
    #     return self._predict(data, ref_features)

    # def _predict_dataloader(self, dataloader, ref_features, class_name, aligned=False):
    #     """This function provides anomaly scores/maps for full dataloaders."""
    #     _ = self.forward_modules.eval()
        
    #     img_scores, pix_scores = [], []
    #     label_list, gt_mask_list = [], []
    #     with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
    #         for idx, data_item in enumerate(data_iterator):
    #             image, label, mask, _ = data_item
    #             gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
    #             label_list.append(label.cpu().numpy().astype(bool).ravel())
                    
    #             _scores, _masks = self._predict(image, ref_features)
    #             for score, mask in zip(_scores, _masks):
    #                 img_scores.append(score)
    #                 pix_scores.append(mask)
    #     return img_scores, pix_scores, label_list, gt_mask_list
    def _predict_dataloader(
        self,
        dataloader,
        ref_features,
        class_name,
        aligned=False,
        return_features=False,
    ):
        """Provides anomaly scores/maps for full dataloaders."""

        _ = self.forward_modules.eval()

        img_scores = []
        pix_scores = []

        label_list = []
        gt_mask_list = []

        feature_list = []
        patch_shape_list = []
        image_path_list = []

        with tqdm.tqdm(
            dataloader,
            desc="Inferring...",
            leave=False
        ) as data_iterator:

            for idx, data_item in enumerate(
                data_iterator
            ):

                # 기존의 _ 대신 path를 받음
                image, label, mask, image_path = data_item

                gt_mask_list.append(
                    mask.squeeze(1)
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                label_list.append(
                    label.cpu()
                    .numpy()
                    .astype(bool)
                    .ravel()
                )

                # image path 저장
                if isinstance(
                    image_path,
                    (list, tuple)
                ):
                    image_path_list.extend(
                        image_path
                    )
                else:
                    image_path_list.append(
                        image_path
                    )

                if return_features:

                    (
                        _scores,
                        _masks,
                        _features,
                        _patch_shape,
                    ) = self._predict(
                        image,
                        ref_features,
                        return_features=True,
                    )

                    feature_list.append(
                        _features
                    )

                    patch_shape_list.append(
                        _patch_shape
                    )

                else:

                    _scores, _masks = self._predict(
                        image,
                        ref_features,
                    )

                for score, mask_item in zip(
                    _scores,
                    _masks
                ):
                    img_scores.append(
                        score
                    )

                    pix_scores.append(
                        mask_item
                    )

        if return_features:

            return (
                img_scores,
                pix_scores,
                label_list,
                gt_mask_list,
                feature_list,
                patch_shape_list,
                image_path_list,
            )

        return (
            img_scores,
            pix_scores,
            label_list,
            gt_mask_list,
        )

    # def _predict(self, images, ref_features):
    #     """Infer score and mask for a batch of images."""
    #     images = images.to(torch.float).to(self.device)
    #     _ = self.forward_modules.eval()

    #     batchsize = images.shape[0]
    #     with torch.no_grad():
    #         features, patch_shapes = self._embed(images, ref_features, provide_patch_shapes=True)
    #         features = np.asarray(features)

    #         patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
    #         image_scores = self.patch_maker.unpatch_scores(
    #             image_scores, batchsize=batchsize
    #         )
    #         image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    #         image_scores = self.patch_maker.score(image_scores)

    #         patch_scores = self.patch_maker.unpatch_scores(
    #             patch_scores, batchsize=batchsize
    #         )
    #         scales = patch_shapes[0]
    #         patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

    #         masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

    #     return [score for score in image_scores], [mask for mask in masks]
    def _predict(
        self,
        images,
        ref_features,
        return_features=False,
    ):
        """Infer score and mask for a batch of images."""

        images = images.to(
            torch.float
        ).to(
            self.device
        )

        _ = self.forward_modules.eval()

        batchsize = images.shape[0]

        with torch.no_grad():

            features, patch_shapes = self._embed(
                images,
                ref_features,
                provide_patch_shapes=True
            )

            features = np.asarray(
                features
            )

            patch_scores = image_scores = (
                self.anomaly_scorer.predict(
                    [features]
                )[0]
            )

            image_scores = (
                self.patch_maker.unpatch_scores(
                    image_scores,
                    batchsize=batchsize
                )
            )

            image_scores = image_scores.reshape(
                *image_scores.shape[:2],
                -1
            )

            image_scores = self.patch_maker.score(
                image_scores
            )

            patch_scores = (
                self.patch_maker.unpatch_scores(
                    patch_scores,
                    batchsize=batchsize
                )
            )

            scales = patch_shapes[0]

            patch_scores = patch_scores.reshape(
                batchsize,
                scales[0],
                scales[1]
            )

            masks = (
                self.anomaly_segmentor
                .convert_to_segmentation(
                    patch_scores
                )
            )

            scores_out = [
                score
                for score in image_scores
            ]

            masks_out = [
                mask
                for mask in masks
            ]

            if return_features:

                return (
                    scores_out,
                    masks_out,
                    features,
                    scales,
                )

            return (
                scores_out,
                masks_out,
            )
    def predict_raw_patch_scores(
        self,
        images,
        ref_features,
    ):
        """
        Return raw PatchCore nearest-neighbor anomaly scores
        before pixel-map interpolation.

        Returns:
            patch_scores: np.ndarray
                Shape: (B, H_patch, W_patch)
        """

        images = images.to(
            torch.float
        ).to(
            self.device
        )

        _ = self.forward_modules.eval()

        batchsize = images.shape[0]

        with torch.no_grad():

            # Same feature extraction used by normal PatchCore inference
            features, patch_shapes = self._embed(
                images,
                ref_features,
                provide_patch_shapes=True,
            )

            features = np.asarray(
                features
            )

            # ====================================================
            # Raw PatchCore patch anomaly distances
            # ====================================================

            patch_scores = self.anomaly_scorer.predict(
                [features]
            )[0]

            # (B * num_patches,) -> (B, num_patches)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores,
                batchsize=batchsize,
            )

            # Spatial patch grid
            h_patch, w_patch = patch_shapes[0]

            patch_scores = patch_scores.reshape(
                batchsize,
                h_patch,
                w_patch,
            )

        return patch_scores

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
    

def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
