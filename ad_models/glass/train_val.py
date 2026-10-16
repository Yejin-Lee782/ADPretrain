import logging
import os
import json
from datetime import datetime
import pandas as pd
from .utils import fix_seeds, create_storage_folder, compute_and_store_final_results

LOGGER = logging.getLogger(__name__)

def train_val(
    encoder,
    projectors,
    dataloaders_list,
    ref_features,
    class_names,
    device,
    with_pretrained,
    results_path='./ad_models/glass/results',
    seed=0,
    log_group='group',
    log_project='project',
    run_name='test',
    test='ckpt',
):
    run_save_path = create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    result_collect = []
    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)
    img_aucs, img_aps, pix_aucs, pix_aps, pix_aupros = [], [], [], [], []
    all_metrics = {}
    for dataloader_count, (dataloaders, class_name) in enumerate(zip(dataloaders_list, class_names)):
        fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        imagesize = (3, 224, 224)
        if with_pretrained:
            glass = get_glass(encoder, projectors, device, imagesize, with_pretrained, pre_proj=0, noise=0.005)  # residual features don't need adapter and need smaller noise
        else:
            glass = get_glass(encoder, projectors, device, imagesize, with_pretrained, pre_proj=1, noise=0.015)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        flag = 0., 0., 0., 0., 0., -1.
        if glass.backbone.seed is not None:
            fix_seeds(glass.backbone.seed, device)

        glass.set_model_dir(os.path.join(models_dir, f"backbone_{0}"), dataset_name)
        if test == 'ckpt':
            metrics = glass.trainer(dataloaders["training"], dataloaders["testing"], ref_features[class_name] if with_pretrained else None, dataset_name)
            img_auc, img_ap, pix_auc, pix_ap, pix_aupro = metrics
            img_aucs.append(img_auc)
            img_aps.append(img_ap)
            pix_aucs.append(pix_auc)
            pix_aps.append(pix_ap)
            pix_aupros.append(pix_aupro)
            print("Class Name: {}, Image AUC | AP | {} | {}, Pixel AUC | AP | AUPRO: {} | {} | {}".format(
                    class_name, img_auc, img_ap, pix_auc, pix_ap, pix_aupro))
            
            all_metrics[class_name] = {'img_auc': img_auc, 'img_ap': img_ap, 'pix_auc': pix_auc, 'pix_ap': pix_ap, 'pix_aupro': pix_aupro}
            save_name = dataset_name.split("_")[0]
            with open(f"{save_name}.json", 'w') as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=4)
            # if type(flag) == int:
            #     row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
            #     df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])

        # if type(flag) != int:
        #     i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = glass.tester(dataloaders["testing"], ref_features, dataset_name)
        #     result_collect.append(
        #         {
        #             "dataset_name": dataset_name,
        #             "image_auroc": i_auroc,
        #             "image_ap": i_ap,
        #             "pixel_auroc": p_auroc,
        #             "pixel_ap": p_ap,
        #             "pixel_pro": p_pro,
        #             "best_epoch": epoch,
        #         }
        #     )

        #     if epoch > -1:
        #         for key, item in result_collect[-1].items():
        #             if isinstance(item, str):
        #                 continue
        #             elif isinstance(item, int):
        #                 print(f"{key}:{item}")
        #             else:
        #                 print(f"{key}:{round(item * 100, 2)} ", end="")

        #     # save results csv after each category
        #     print("\n")
        #     result_metric_names = list(result_collect[-1].keys())[1:]
        #     result_dataset_names = [results["dataset_name"] for results in result_collect]
        #     result_scores = [list(results.values())[1:] for results in result_collect]
        #     compute_and_store_final_results(
        #         run_save_path,
        #         result_scores,
        #         result_metric_names,
        #         row_names=result_dataset_names,
        #     )

    # # save distribution judgment xlsx after all categories
    # if len(df['Class']) != 0:
    #     os.makedirs('./datasets/excel', exist_ok=True)
    #     xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
    #     df.to_excel(xlsx_path, index=False)
    return img_aucs, img_aps, pix_aucs, pix_aps, pix_aupros


def get_glass(
    backbone,
    projectors,
    device,
    input_shape,
    with_pretrained,
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    patchsize=3,
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
):
    backbone_seed = None
    backbone.seed = backbone_seed

    from .glass import GLASS
    glass_inst = GLASS(device)
    glass_inst.load(
        backbone=backbone,
        projectors=projectors,
        device=device,
        input_shape=input_shape,
        with_pretrained=with_pretrained,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        meta_epochs=meta_epochs,
        eval_epochs=eval_epochs,
        dsc_layers=dsc_layers,
        dsc_hidden=dsc_hidden,
        dsc_margin=dsc_margin,
        train_backbone=train_backbone,
        pre_proj=pre_proj,
        mining=mining,
        noise=noise,
        radius=radius,
        p=p,
        lr=lr,
        svd=svd,
        step=step,
        limit=limit,
    )
    
    return glass_inst