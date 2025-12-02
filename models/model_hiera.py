from functools import partial
from pathlib import Path

from hiera import Head, Hiera, MaskedAutoencoderHiera
import torch
from torch import nn

try:
    from experiment_config import config
except ImportError:
    import sys

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from experiment_config import config

from models.model_utils import hack_embeddings_spatial, interpolate_embeddings_spatial, interpolate_embeddings_temporal, get_pretrained_model


class HieraFusionHead(MaskedAutoencoderHiera):
    """
    This model inherits the encoder and multi-scale fusion from MaskedAutoencoderHiera
    but replaces the MAE decoder with a classification head.
    """

    def __init__(self, num_classes, head_dropout, head_init_scale, sep_pos_embed, **kwargs):
        super().__init__(num_classes=num_classes, head_dropout=head_dropout,
                         head_init_scale=head_init_scale, sep_pos_embed=sep_pos_embed, **kwargs)

        # remove all the decoder-specific parts
        del self.decoder_embed
        del self.mask_token
        del self.decoder_pos_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        # add the classification-specific layers
        encoder_dim_out = self.encoder_norm.normalized_shape[0]
        self.norm = nn.LayerNorm(encoder_dim_out, eps=1e-6)
        self.head = Head(encoder_dim_out, num_classes,
                         dropout_rate=head_dropout)

        # initialize everything (as done in Hiera)
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

    def forward(self, x):
        x, _ = self.forward_encoder(x, mask_ratio=0.0)
        # flatten all dimensions except the first (batch) and the last (channels)
        x = x.flatten(start_dim=1, end_dim=-2)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x


class Hiera2D(nn.Module):
    def __init__(self, image_size, kind="pretrained", num_classes=1, patch_size=4):
        super().__init__()
        assert kind in ["pretrained",
                        "finetuned"], f"Unknown model kind: {kind}"
        model = HieraFusionHead if config.FUSION_HEAD_ENABLED else Hiera
        self.hiera = model(
            num_classes=num_classes,
            input_size=(image_size, image_size),
            drop_path_rate=config.DROP_PATH_RATE,
            head_dropout=config.HEAD_DROPOUT,
            head_init_scale=config.HEAD_INIT_SCALE,
            q_pool=2 if config.FUSION_HEAD_ENABLED else 3
        )
        # disable softmax in head for eval and testing
        self.hiera.head.act_func = nn.Identity()

        if kind == "pretrained":
            state_dict = get_pretrained_model()
            if "model_state" in state_dict:
                state_dict = state_dict["model_state"]

            new_state_dict = interpolate_embeddings_spatial(
                image_size=image_size,
                patch_size=patch_size,
                model_state=state_dict,
                pos_embedding_key="pos_embed"
            )

            self.hiera.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        return self.hiera(x)


class Hiera3D(nn.Module):
    def __init__(self, image_size, image_depth, kind="pretrained", num_classes=1, patch_size=4, patch_depth=2):
        super().__init__()
        assert kind in ["pretrained",
                        "finetuned"], f"Unknown model kind: {kind}"
        model = HieraFusionHead if config.FUSION_HEAD_ENABLED else Hiera
        self.hiera = model(
            embed_dim=config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]["embed_dim"],
            num_heads=config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]["num_heads"],
            stages=config.PRETRAINED_MODEL_CONFIGS[config.PRETRAINED_MODEL]["stages"],
            num_classes=num_classes,
            input_size=(image_depth, image_size, image_size),
            q_stride=(1, 2, 2),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(3, 7, 7),
            patch_stride=(2, 4, 4),
            patch_padding=(1, 3, 3),
            sep_pos_embed=True,
            drop_path_rate=config.DROP_PATH_RATE,
            head_dropout=config.HEAD_DROPOUT,
            head_init_scale=config.HEAD_INIT_SCALE,
            q_pool=2 if config.FUSION_HEAD_ENABLED else 3
        )
        # disable softmax in head for eval and testing
        self.hiera.head.act_func = nn.Identity()

        if kind == "pretrained":
            state_dict = get_pretrained_model()
            if "model_state" in state_dict:
                state_dict = state_dict["model_state"]

            if config.HACK["ENABLE"]:
                new_state_dict = hack_embeddings_spatial(
                    image_size=image_size,
                    patch_size=patch_size,
                    model_state=state_dict)
            else:
                new_state_dict = interpolate_embeddings_spatial(
                    image_size=image_size,
                    patch_size=patch_size,
                    model_state=state_dict
                )
            new_state_dict = interpolate_embeddings_temporal(
                image_depth=image_depth,
                patch_depth=patch_depth,
                model_state=new_state_dict
            )

            self.hiera.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        return self.hiera(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification


class VideoMAESmallImageAdapter(nn.Module):
    """
    Adapts a pretrained VideoMAE model to work with smaller input images.
    
    Args:
        model_name: HuggingFace model name or path
        new_image_size: Target image size (e.g., 64 for 64x64 images)
        num_classes: Number of output classes (default: 2 for binary classification)
    """
    
    def __init__(
        self, 
        model_name="MCG-NJU/videomae-large-finetuned-kinetics",
        new_image_size=64,
        num_classes=2
    ):
        super().__init__()
        
        # Load pretrained model
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        
        # Get config values
        patch_size = self.model.config.patch_size
        tubelet_size = self.model.config.tubelet_size
        old_image_size = self.model.config.image_size
        
        # Update config for new image size
        self.model.config.image_size = new_image_size
        self.model.videomae.embeddings.patch_embeddings.image_size = (new_image_size, new_image_size)
        
        # Calculate patch dimensions
        num_frames = 16
        new_h_patches = new_image_size // patch_size
        new_w_patches = new_image_size // patch_size
        new_t_patches = num_frames // tubelet_size
        new_num_patches = new_t_patches * new_h_patches * new_w_patches
        
        old_h_patches = old_image_size // patch_size
        old_w_patches = old_image_size // patch_size
        old_spatial_patches = old_h_patches * old_w_patches
        
        # Interpolate position embeddings
        old_pos_embed = self.model.videomae.embeddings.position_embeddings.data
        embed_dim = old_pos_embed.shape[-1]
        old_num_patches = old_pos_embed.shape[1] - 1  # Exclude CLS
        old_t_patches = old_num_patches // old_spatial_patches
        
        # Extract and reshape patch embeddings
        patch_pos_embed = old_pos_embed[:, 1:, :]
        usable_patches = old_t_patches * old_spatial_patches
        patch_pos_embed = patch_pos_embed[:, :usable_patches, :]
        patch_pos_embed = patch_pos_embed.reshape(1, old_t_patches, old_h_patches, old_w_patches, embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)
        
        # Interpolate to new size
        new_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_t_patches, new_h_patches, new_w_patches),
            mode='trilinear',
            align_corners=False
        )
        
        new_patch_pos_embed = new_patch_pos_embed.permute(0, 2, 3, 4, 1)
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, new_num_patches, embed_dim)
        
        # Update position embeddings
        self.model.videomae.embeddings.position_embeddings = nn.Parameter(new_patch_pos_embed)
        
        # Replace classifier
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(nn.LayerNorm(hidden_size, eps=1e-6),
                                            nn.Linear(hidden_size, num_classes))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            pixel_values: Video tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.model(x)
        return outputs.logits

if __name__ == "__main__":

    # Example usage:
    # python -m hiera-luna25-finetuning.models.model_hiera

    IMG_SIZE, IMG_DEPTH, MODE = 64, 16, "3D"
    print(f"{MODE} Hiera:")
    if MODE == "3D":
        model = Hiera3D(image_size=IMG_SIZE, image_depth=IMG_DEPTH)
        print(
            f"Output shape: {model(torch.randn(1, 3, IMG_DEPTH, IMG_SIZE, IMG_SIZE)).shape}")
    else:
        model = Hiera2D(image_size=IMG_SIZE)
        print(
            f"Output shape: {model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")
        else:
            print(f"[Frozen] {name}: {param.size()}")