import logging
import os

import numpy as np
import torch
from torchvision.transforms import v2

import dataloader
from models.model_hiera import Hiera3D, VideoMAE, VJEPA2


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


class MalignancyProcessor:
    """
    Loads a chest CT scan and predicts malignancy around nodules
    using an ensemble of 3 models:
      - Hiera3D
      - VideoMAE
      - VJEPA2
    """

    def __init__(self, mode='3D', suppress_logs=False, ensemble_weights=None):
        self.size_px = 64
        self.size_mm = 50
        self.depth_px = 16

        self.mode = "3D"
        self.suppress_logs = suppress_logs
        self.model_root = "./results/"

        # Optional weights for weighted ensemble.
        # If None, equal averaging is used.
        self.ensemble_weights = ensemble_weights

        if not self.suppress_logs:
            logging.info("Initializing ensemble deep learning system")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define checkpoint folders under /opt/app/resources/
        # Change these folder names if your checkpoint directories differ.
        self.hiera_dir = "finetune-hiera-3D-20260322/fold_1"
        self.videomae_dir = "finetune-VideoMAE-20260322/fold_1"
        self.vjepa2_dir = "finetune-VJEPA2-20260320/fold_1"

        # Build models
        self.model_hiera3d = Hiera3D(
            image_size=self.size_px,
            image_depth=self.depth_px,
            kind="finetuned",
        ).to(self.device)

        self.model_videomae = VideoMAE(
            model_name="MCG-NJU/videomae-large-finetuned-kinetics",
            new_image_size=64,
            num_classes=1,
        ).to(self.device)

        self.model_vjepa2 = VJEPA2(
            model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
            new_image_size=64,
            num_classes=1,
        ).to(self.device)

        self._load_checkpoints()

    def _load_single_checkpoint(self, model, checkpoint_dir):
        ckpt_path = os.path.join(
            self.model_root,
            checkpoint_dir,
            "best_metric_model.pth",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt)
        model.eval()

        if not self.suppress_logs:
            logging.info("Loaded checkpoint: %s", ckpt_path)

    def _load_checkpoints(self):
        self._load_single_checkpoint(self.model_hiera3d, self.hiera_dir)
        self._load_single_checkpoint(self.model_videomae, self.videomae_dir)
        self._load_single_checkpoint(self.model_vjepa2, self.vjepa2_dir)

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):
        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        patch = patch.astype(np.float32)
        patch = dataloader.clip_and_scale(patch)

        normalize_transform = v2.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        )

        patch = normalize_transform(torch.from_numpy(patch))

        # Expected 3D format: (D, C, H, W) -> (C, D, H, W)
        patch = patch.permute(1, 0, 2, 3)
        patch = patch.detach().cpu().numpy()

        return patch

    def _prepare_nodules(self):
        if not self.suppress_logs:
            logging.info("Extracting 3D patches")

        output_shape = [self.depth_px, self.size_px, self.size_px]
        nodules = []

        for _coord in self.coords:
            patch = self.extract_patch(_coord, output_shape, mode="3D")
            nodules.append(patch)

        nodules = np.array(nodules, dtype=np.float32)
        nodules = torch.from_numpy(nodules).to(self.device)
        return nodules

    def _ensemble_logits(self, logits_list):
        """
        logits_list: list of tensors, each shape [N] or [N, 1]
        returns: ensemble logits tensor shape [N]
        """
        processed = [x.reshape(-1) for x in logits_list]
        stacked = torch.stack(processed, dim=0)  # [3, N]

        if self.ensemble_weights is None:
            ensemble_logits = stacked.mean(dim=0)
        else:
            weights = torch.tensor(
                self.ensemble_weights,
                dtype=stacked.dtype,
                device=stacked.device,
            )
            weights = weights / weights.sum()
            ensemble_logits = (stacked * weights.view(-1, 1)).sum(dim=0)

        return ensemble_logits

    def _process_ensemble(self):
        if not self.suppress_logs:
            logging.info("Running ensemble inference")

        nodules = self._prepare_nodules()

        with torch.no_grad():
            logits_hiera3d = self.model_hiera3d(nodules).reshape(-1)
            logits_videomae = self.model_videomae(nodules).reshape(-1)
            logits_vjepa2 = self.model_vjepa2(nodules).reshape(-1)

            ensemble_logits = self._ensemble_logits(
                [logits_hiera3d, logits_videomae, logits_vjepa2]
            )

        result = {
            "ensemble_logits": ensemble_logits.detach().cpu().numpy(),
            "hiera3d_logits": logits_hiera3d.detach().cpu().numpy(),
            "videomae_logits": logits_videomae.detach().cpu().numpy(),
            "vjepa2_logits": logits_vjepa2.detach().cpu().numpy(),
        }
        return result

    def predict(self):
        outputs = self._process_ensemble()

        ensemble_logits = outputs["ensemble_logits"]
        ensemble_prob = torch.sigmoid(torch.from_numpy(ensemble_logits)).numpy()

        per_model_logits = {
            "hiera3d_logits": outputs["hiera3d_logits"],
            "videomae_logits": outputs["videomae_logits"],
            "vjepa2_logits": outputs["vjepa2_logits"],
        }
        # print(per_model_logits)
        return ensemble_prob, ensemble_logits
