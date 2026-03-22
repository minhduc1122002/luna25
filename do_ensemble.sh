python ensemble.py \
  --eval_csv ./data/data_splits/val_fold_1.csv \
  --hiera_ckpt ./results/finetune-hiera-3D-20260322/fold_1/best_metric_model.pth \
  --videomae_ckpt ./results/finetune-VideoMAE-20260322/fold_1/best_metric_model.pth \
  --vjepa2_ckpt ./results/finetune-VJEPA2-20260320/fold_1/best_metric_model.pth \
  --output_dir ./results/ensemble_result \
  --ensemble_weights 0.5 0.25 0.25