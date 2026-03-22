import os
import warnings

try:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
    import torchvision

    torchvision.disable_beta_transforms_warning()
except Exception as e:
    print(f"An error occurred during setup: {e}")

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm

from dataloader import get_data_loader
from experiment_config import config
from models.model_hiera import Hiera3D, VideoMAE, VJEPA2


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(model_name, device):
    model_name = model_name.lower()

    if model_name == "hiera3d":
        model = Hiera3D(
            image_size=config.SIZE_PX,
            image_depth=config.DEPTH_PX,
        ).to(device)

    elif model_name == "videomae":
        model = VideoMAE(
            model_name="MCG-NJU/videomae-large-finetuned-kinetics",
            new_image_size=64,
            num_classes=1,
        ).to(device)

    elif model_name == "vjepa2":
        model = VJEPA2(
            model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
            new_image_size=64,
            num_classes=1,
        ).to(device)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def load_checkpoint_model(model_name, checkpoint_path, device):
    model = build_model(model_name, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Loaded %s from %s", model_name, checkpoint_path)
    return model


def build_eval_loader(eval_csv_path):
    eval_df = pd.read_csv(eval_csv_path)

    logging.info("Evaluation CSV: %s", eval_csv_path)
    logging.info("Total samples: %d", len(eval_df))
    logging.info("Malignant: %d", int(eval_df.label.sum()))
    logging.info("Benign: %d", int(len(eval_df) - eval_df.label.sum()))

    eval_loader = get_data_loader(
        config.DATADIR,
        eval_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.MINI_BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        augmentation=False,
    )
    return eval_df, eval_loader


def predict_ensemble(models, data_loader, device, weights=None):
    """
    Average logits from different architectures, then apply sigmoid.
    """
    y_true_all = []
    y_prob_all = []

    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        weights = weights / weights.sum()

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Ensemble inference"):
            inputs, labels = batch_data["image"], batch_data["label"]
            inputs = inputs.to(device)
            labels = labels.float().to(device).reshape(-1)

            batch_logits = []
            for model in models:
                logits = model(inputs).reshape(-1)
                batch_logits.append(logits)

            batch_logits = torch.stack(batch_logits, dim=0)  # [n_models, batch]

            if weights is None:
                mean_logits = batch_logits.mean(dim=0)
            else:
                mean_logits = (batch_logits * weights.view(-1, 1)).sum(dim=0)

            mean_probs = torch.sigmoid(mean_logits)

            y_true_all.append(labels.cpu())
            y_prob_all.append(mean_probs.cpu())

    y_true = torch.cat(y_true_all, dim=0).numpy().reshape(-1)
    y_prob = torch.cat(y_prob_all, dim=0).numpy().reshape(-1)
    return y_true, y_prob


def save_outputs(eval_df, y_true, y_prob, output_dir, model_descriptions):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    auc_metric = metrics.auc(fpr, tpr)

    y_pred_label = (y_prob >= 0.5).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred_label)
    precision = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)

    logging.info("AUC: %.5f", auc_metric)
    logging.info("ACC: %.5f", acc)
    logging.info("Precision: %.5f", precision)
    logging.info("Recall: %.5f", recall)
    logging.info("F1: %.5f", f1)

    preds_df = pd.DataFrame(
        {
            "AnnotationID": eval_df["AnnotationID"],
            "label": y_true.astype(int),
            "probability": y_prob,
            "prediction": y_pred_label,
        }
    )
    preds_df.to_csv(output_dir / "ensemble_predictions.csv", index=False)

    summary = {
        "auc": float(auc_metric),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "models": model_descriptions,
    }
    np.save(output_dir / "ensemble_metrics.npy", summary)

    logging.info("Saved predictions to %s", output_dir / "ensemble_predictions.csv")
    logging.info("Saved metrics to %s", output_dir / "ensemble_metrics.npy")


def evaluate_ensemble(
    eval_csv_path,
    hiera_ckpt,
    videomae_ckpt,
    vjepa2_ckpt,
    output_dir,
    weights=None,
):
    device = get_device()
    logging.info("Using device: %s", device)

    eval_df, eval_loader = build_eval_loader(eval_csv_path)

    models = [
        load_checkpoint_model("hiera3d", hiera_ckpt, device),
        load_checkpoint_model("videomae", videomae_ckpt, device),
        load_checkpoint_model("vjepa2", vjepa2_ckpt, device),
    ]

    y_true, y_prob = predict_ensemble(
        models=models,
        data_loader=eval_loader,
        device=device,
        weights=weights,
    )

    model_descriptions = [
        {"name": "hiera3d", "checkpoint": str(hiera_ckpt)},
        {"name": "videomae", "checkpoint": str(videomae_ckpt)},
        {"name": "vjepa2", "checkpoint": str(vjepa2_ckpt)},
    ]

    if weights is not None:
        for item, w in zip(model_descriptions, weights):
            item["weight"] = float(w)

    save_outputs(
        eval_df=eval_df,
        y_true=y_true,
        y_prob=y_prob,
        output_dir=output_dir,
        model_descriptions=model_descriptions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble evaluation for 3 different model architectures.")

    parser.add_argument("--eval_csv", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--hiera_ckpt", type=str, required=True, help="Path to Hiera3D checkpoint")
    parser.add_argument("--videomae_ckpt", type=str, required=True, help="Path to VideoMAE checkpoint")
    parser.add_argument("--vjepa2_ckpt", type=str, required=True, help="Path to VJEPA2 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")

    parser.add_argument(
        "--ensemble_weights",
        type=float,
        nargs=3,
        default=None,
        help="Optional ensemble weights, e.g. --ensemble_weights 0.4 0.3 0.3",
    )

    args = parser.parse_args()

    evaluate_ensemble(
        eval_csv_path=args.eval_csv,
        hiera_ckpt=args.hiera_ckpt,
        videomae_ckpt=args.videomae_ckpt,
        vjepa2_ckpt=args.vjepa2_ckpt,
        output_dir=args.output_dir,
        weights=args.ensemble_weights,
    )