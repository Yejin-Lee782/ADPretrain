import os
import argparse
import numpy as np
from PIL import Image

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from models.imagebind import ImageBindModel
from models.clip import ClipModel
from models.dino import DinoModel
from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD


class FEWSHOTDATA(Dataset):
    
    def __init__(self, 
                 root: str,
                 class_name: str = 'bottle', 
                 train: bool = True,
                 normalize='imagebind',
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.mask_size = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
    
        if normalize == "imagebind":
            self.transform = T.Compose(  # for imagebind
                [
                    T.Resize(
                        kwargs.get('img_size', 224), interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(kwargs.get('crp_size', 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])
        
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 256), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 256)),
            T.ToTensor()])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, mask_path, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        
        return img, label, mask, class_name
    
    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
       
        img = self.transform(img)
        
        if label == 0:
            mask = torch.zeros([1, self.mask_size[0], self.mask_size[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.train else 'test'
        
        image_dir = os.path.join(self.root, class_name, phase)
        mask_dir = os.path.join(self.root, class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                    
        class_names = [class_name] * len(image_paths)
        return image_paths, labels, mask_paths, class_names
    

def main(args):
    image_size = 224
    device = args.device
    backbone = args.backbone
    dataset = args.dataset

    #root_dir = f'./data/8shot/{dataset}'
    root_dir = args.data_path

    DATASET_CLASSES = {
        'mvtec': MVTEC,
        'visa': VISA,
        'btad': BTAD,
        'mvtec3d': MVTEC3D,
        'mpdd': MPDD,
    }

    if dataset == "weld":
        CLASS_NAMES = ["weld"]
    else:
        CLASS_NAMES = DATASET_CLASSES[dataset].CLASS_NAMES

    # ========================================================
    # Backbone
    # ========================================================

    if backbone == 'imagebind':
        encoder = ImageBindModel(
            backbone,
            device=device
        )
        normalize = 'imagebind'

    elif 'clip' in backbone:
        encoder = ClipModel(
            backbone,
            device=device
        )
        normalize = 'imagebind'

    elif 'dino' in backbone:
        encoder = DinoModel(
            backbone,
            device=device
        )
        normalize = 'imagenet'

    else:
        raise ValueError(
            "Unrecognized Backbone!"
        )

    encoder.to(device)

    encoder.eval()

    feature_dimensions = encoder.feature_dimensions

    # ========================================================
    # Extract references
    # ========================================================

    for class_name in CLASS_NAMES:

        print()
        print("==============================")
        print(f"Dataset   : {dataset}")
        print(f"Class     : {class_name}")
        print(f"Backbone  : {backbone}")
        print(f"Root      : {root_dir}")
        print("==============================")
        print()

        train_dataset = FEWSHOTDATA(
            root_dir,
            class_name=class_name,
            train=True,
            normalize=normalize,
            img_size=image_size,
            crp_size=image_size,
            msk_size=image_size,
            msk_crp_size=image_size,
        )

        print(
            f"Reference images: "
            f"{len(train_dataset)}"
        )

        if len(train_dataset) != 8:
            print(
                f"[WARNING] Expected 8 reference images, "
                f"but found {len(train_dataset)}."
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        layer1_features = []
        layer2_features = []
        layer3_features = []
        layer4_features = []

        for batch in tqdm.tqdm(train_loader):

            images, _, _, _ = batch

            with torch.no_grad():

                features = (
                    encoder.encode_image_from_tensors(
                        images.to(device)
                    )
                )

            layer1_features.append(
                features[0]
            )

            layer2_features.append(
                features[1]
            )

            layer3_features.append(
                features[2]
            )

            layer4_features.append(
                features[3]
            )

        layer1_features = torch.cat(
            layer1_features,
            dim=0
        )

        layer2_features = torch.cat(
            layer2_features,
            dim=0
        )

        layer3_features = torch.cat(
            layer3_features,
            dim=0
        )

        layer4_features = torch.cat(
            layer4_features,
            dim=0
        )

        print()
        print("Before flatten:")
        print("Layer1:", layer1_features.shape)
        print("Layer2:", layer2_features.shape)
        print("Layer3:", layer3_features.shape)
        print("Layer4:", layer4_features.shape)

        # B,C,H,W
        # ↓
        # B,H,W,C
        # ↓
        # (B*H*W), C

        layer1_features = (
            layer1_features
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                feature_dimensions[0]
            )
        )

        layer2_features = (
            layer2_features
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                feature_dimensions[1]
            )
        )

        layer3_features = (
            layer3_features
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                feature_dimensions[2]
            )
        )

        layer4_features = (
            layer4_features
            .permute(0, 2, 3, 1)
            .reshape(
                -1,
                feature_dimensions[3]
            )
        )

        print()
        print("After flatten:")
        print("Layer1:", layer1_features.shape)
        print("Layer2:", layer2_features.shape)
        print("Layer3:", layer3_features.shape)
        print("Layer4:", layer4_features.shape)

        save_dir = (
            f'ref_features/'
            f'{backbone}/'
            f'{dataset}_8shot'
        )

        class_save_dir = os.path.join(
            save_dir,
            class_name
        )

        os.makedirs(
            class_save_dir,
            exist_ok=True
        )

        np.save(
            os.path.join(
                class_save_dir,
                'layer1.npy'
            ),
            layer1_features.cpu().numpy()
        )

        np.save(
            os.path.join(
                class_save_dir,
                'layer2.npy'
            ),
            layer2_features.cpu().numpy()
        )

        np.save(
            os.path.join(
                class_save_dir,
                'layer3.npy'
            ),
            layer3_features.cpu().numpy()
        )

        np.save(
            os.path.join(
                class_save_dir,
                'layer4.npy'
            ),
            layer4_features.cpu().numpy()
        )

        print()
        print(
            f"Reference features saved to:"
        )
        print(
            f"  {class_save_dir}"
        )
        
        
# if __name__ == '__main__':
#     TOTAL_SHOT = 8
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--backbone', type=str, default="dinov2-large")
#     parser.add_argument('--dataset', type=str, default="mvtec")
#     parser.add_argument('--device', type=str, default="cuda:0")
   
#     args = parser.parse_args()
    
#     main(args)
if __name__ == '__main__':

    TOTAL_SHOT = 8

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--backbone',
        type=str,
        default="dinov2-base"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default="weld"
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="/ssdg/spl_yejin/aihub_weld/weld_8shot"
    )

    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0"
    )

    args = parser.parse_args()

    main(args)