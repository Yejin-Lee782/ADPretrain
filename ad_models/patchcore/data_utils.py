import os
import pandas


MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

BTAD_CLASS_NAMES = ['01', '02', '03']

MVTEC3D_CLASS_NAMES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
               'foam', 'peach', 'potato', 'rope', 'tire']

MPDD_CLASS_NAMES = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']


def get_normal_image_paths_mvtec(root, class_name):
    phase = 'train' 
    image_paths = []

    image_dir = os.path.join(root, class_name, phase)
    
    img_types = sorted(os.listdir(image_dir))
    for img_type in img_types:
        # load images
        img_type_dir = os.path.join(image_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
        image_paths.extend(img_fpath_list)

    return image_paths


def get_normal_image_paths_visa(root, class_name):
    split_csv_file = os.path.join(root, 'split_csv', '1cls.csv')
    csv_data = pandas.read_csv(split_csv_file)
    class_data = csv_data.loc[csv_data['object'] == class_name]
    
    train_data = class_data.loc[class_data['split'] == 'train']
    image_paths = train_data['image'].to_list()
    image_paths = [os.path.join(root, file_name) for file_name in image_paths]

    return image_paths