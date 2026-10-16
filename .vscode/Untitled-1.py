        # ====================================================
        # Save training checkpoint BEFORE evaluation
        # ====================================================

        os.makedirs(
            args.checkpoint_path,
            exist_ok=True,
        )

        train_checkpoint = {
            "epoch": epoch + 1,
            "projectors": projectors.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "backbone": args.backbone,
        }

        train_ckpt_path = os.path.join(
            args.checkpoint_path,
            f"weld_epoch_{epoch + 1:02d}_before_eval.pth",
        )

        torch.save(
            train_checkpoint,
            train_ckpt_path,
        )

        print(
            f"\n[Checkpoint] Training state saved: "
            f"{train_ckpt_path}"
        )


        # ====================================================
        # Welding Evaluation
        # ====================================================

        if (epoch + 1) % args.eval_freq == 0:

            print()
            print("============================================")
            print(f" Welding Evaluation - Epoch {epoch + 1}")
            print("============================================")

            projectors.eval()


            # ------------------------------------------------
            # Evaluation normal reference features
            #
            # ref_features/dinov2-base/weld_8shot/weld/
            #   layer1.npy
            #   layer2.npy
            #   layer3.npy
            #   layer4.npy
            # ------------------------------------------------

            ref_features = load_mc_reference_features(
                args.test_ref_feature_dir,
                ["weld"],
                args.device,
                args.num_ref_shot,
            )

            weld_ref_features = ref_features["weld"]

            print("\nEvaluation reference features:")

            for i, ref in enumerate(
                weld_ref_features,
                start=1,
            ):
                print(
                    f"  Layer {i}: {tuple(ref.shape)}"
                )


            # ------------------------------------------------
            # PatchCore normal memory bank
            #
            # /weld_eval_root/weld/train/good
            # ------------------------------------------------

            normal_dataset = MVTEC(
                args.test_dataset_dir,
                class_name="weld",
                train=True,
                normalize=NORMALIZE,
                img_size=224,
                crp_size=224,
                msk_size=224,
                msk_crp_size=224,
            )


            # ------------------------------------------------
            # Test:
            #
            # /weld_eval_root/weld/test/
            #     good
            #     porosity
            #     incomplete_penetration
            #     lack_of_fusion
            #     undercut
            # ------------------------------------------------

            test_dataset = MVTEC(
                args.test_dataset_dir,
                class_name="weld",
                train=False,
                normalize=NORMALIZE,
                img_size=224,
                crp_size=224,
                msk_size=224,
                msk_crp_size=224,
            )


            print(
                f"Normal memory-bank images: "
                f"{len(normal_dataset)}"
            )

            print(
                f"Evaluation images: "
                f"{len(test_dataset)}"
            )


            normal_loader = DataLoader(
                normal_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                drop_last=False,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                drop_last=False,
            )


            # ------------------------------------------------
            # ADPretrain original validate()
            # ------------------------------------------------

            metrics = validate(
                args,
                encoder,
                projectors,
                normal_loader,
                test_loader,
                weld_ref_features,
                args.device,
                "weld",
                residual=True,
                pro=True,
                dtype=DTYPE,
            )


            # =================================================
            # PatchCore / Angle metrics
            # =================================================

            (
                img_auc,
                img_ap,
                img_f1_score,
                pix_auc,
                pix_ap,
                pix_f1_score,
                pix_aupro,
            ) = metrics["metrics"]


            print()
            print("--------------------------------------------")
            print("[PatchCore / Angle]")
            print(f"Image AUC : {img_auc:.5f}")
            print(f"Image AP  : {img_ap:.5f}")
            print(f"Image F1  : {img_f1_score:.5f}")
            print(f"Pixel AUC : {pix_auc:.5f}")
            print(f"Pixel AP  : {pix_ap:.5f}")
            print(f"Pixel F1  : {pix_f1_score:.5f}")
            print(f"AUPRO     : {pix_aupro:.5f}")
            print("--------------------------------------------")


            state_dict = {
                "epoch": epoch + 1,
                "projectors": projectors.state_dict(),
                "img_auc": img_auc,
                "pix_auc": pix_auc,
                "pix_pro": pix_aupro,
                "backbone": args.backbone,
            }


            if img_auc > best_img_auc:

                best_img_auc = img_auc

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_img_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best Image-AUC checkpoint"
                )


            if pix_auc > best_pix_auc:

                best_pix_auc = pix_auc

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pix_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best Pixel-AUC checkpoint"
                )


            if pix_aupro > best_pro:

                best_pro = pix_aupro

                torch.save(
                    state_dict,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pro_angle.pth",
                    ),
                )

                print(
                    "[SAVE] New best AUPRO checkpoint"
                )


            # =================================================
            # Feature Norm metrics
            # =================================================

            (
                img_auc_norm,
                img_ap_norm,
                img_f1_norm,
                pix_auc_norm,
                pix_ap_norm,
                pix_f1_norm,
                pix_aupro_norm,
            ) = metrics["metrics-norm"]


            print()
            print("--------------------------------------------")
            print("[Feature Norm]")
            print(f"Image AUC : {img_auc_norm:.5f}")
            print(f"Image AP  : {img_ap_norm:.5f}")
            print(f"Image F1  : {img_f1_norm:.5f}")
            print(f"Pixel AUC : {pix_auc_norm:.5f}")
            print(f"Pixel AP  : {pix_ap_norm:.5f}")
            print(f"Pixel F1  : {pix_f1_norm:.5f}")
            print(f"AUPRO     : {pix_aupro_norm:.5f}")
            print("--------------------------------------------")


            state_dict_norm = {
                "epoch": epoch + 1,
                "projectors": projectors.state_dict(),
                "img_auc": img_auc_norm,
                "pix_auc": pix_auc_norm,
                "pix_pro": pix_aupro_norm,
                "backbone": args.backbone,
            }


            if img_auc_norm > best_img_auc_norm:

                best_img_auc_norm = img_auc_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_img_norm.pth",
                    ),
                )


            if pix_auc_norm > best_pix_auc_norm:

                best_pix_auc_norm = pix_auc_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pix_norm.pth",
                    ),
                )


            if pix_aupro_norm > best_pro_norm:

                best_pro_norm = pix_aupro_norm

                torch.save(
                    state_dict_norm,
                    os.path.join(
                        args.checkpoint_path,
                        "checkpoints_pro_norm.pth",
                    ),
                )


            print()
            print("============================================")
            print(
                f"Best Image AUC : {best_img_auc:.5f}"
            )
            print(
                f"Best Pixel AUC : {best_pix_auc:.5f}"
            )
            print(
                f"Best AUPRO     : {best_pro:.5f}"
            )
            print("============================================")