from functools import partial
from pathlib import Path

from hiera import Head, Hiera, MaskedAutoencoderHiera
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEModel, VJEPA2Model
import math
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

class VideoMAE(nn.Module):
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
        self.model = VideoMAEModel.from_pretrained(model_name)
        # self.model = TimesformerModel.from_pretrained(model_name)
        patch_size = self.model.config.patch_size
        tubelet_size = self.model.config.tubelet_size
        old_image_size = self.model.config.image_size
        
        # Update config for new image size
        self.model.config.image_size = new_image_size
        self.model.embeddings.patch_embeddings.image_size = (new_image_size, new_image_size)
        
        # Interpolate position embeddings using video-specific method
        old_pos_embed = self.model.embeddings.position_embeddings.data
        new_pos_embed = self._interpolate_video_embeddings(
            old_image_size, new_image_size, patch_size, tubelet_size, 
            16, old_pos_embed, 'bicubic'
        )
        
        # Concatenate CLS and patches
        # new_pos_embed = torch.cat([cls_pos_embed, new_patch_pos_embed], dim=1)
        self.model.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        
        # Replace classifier
        hidden_size = self.model.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def _interpolate_video_embeddings(
        self, old_image_size, new_image_size, patch_size, tubelet_size,
        num_frames, pos_embedding, interpolation_mode
    ):
        """
        Interpolate position embeddings for video by separating temporal and spatial dimensions.
        """
        n, seq_length, hidden_dim = pos_embedding.shape
        
        # Calculate spatial patches per frame
        old_spatial_patches = (old_image_size // patch_size) ** 2
        
        # Calculate temporal patches
        old_t_patches = seq_length // old_spatial_patches
        
        # Use only complete frames
        usable_patches = old_t_patches * old_spatial_patches
        pos_embedding = pos_embedding[:, :usable_patches, :]
        
        # Reshape to separate temporal frames: (1, T*H*W, D) -> (T, H*W, D)
        pos_embedding = pos_embedding.reshape(old_t_patches, old_spatial_patches, hidden_dim)
        
        # Apply spatial interpolation to each temporal frame
        interpolated_frames = []
        for t in range(old_t_patches):
            # Get one frame: (H*W, D) -> (1, H*W, D)
            frame_embed = pos_embedding[t:t+1, :, :]
            
            # Apply the spatial interpolation function
            interpolated_frame = self._interpolate_embeddings_spatial(
                new_image_size, patch_size, frame_embed, interpolation_mode
            )
            
            interpolated_frames.append(interpolated_frame)
        
        # Stack frames back: T x (1, H*W, D) -> (1, T*H*W, D)
        return torch.cat(interpolated_frames, dim=1)
    
    def _interpolate_embeddings_spatial(
        self,
        image_size: int,
        patch_size: int,
        pos_embedding: torch.Tensor,
        interpolation_mode: str = "bicubic"
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for pretrained models.
        Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
        """
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        new_seq_length = (image_size // patch_size) ** 2

        if new_seq_length != seq_length:
            pos_embedding = pos_embedding.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            if seq_length_1d * seq_length_1d != seq_length:
                raise ValueError(
                    f"seq_length is not a perfect square! Instead got "
                    f"seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d} "
                    f"and seq_length = {seq_length}"
                )

            pos_embedding = pos_embedding.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = image_size // patch_size

            new_pos_embedding = nn.functional.interpolate(
                pos_embedding,
                size=new_seq_length_1d,
                mode=interpolation_mode,
                align_corners=True
            )

            new_pos_embedding = new_pos_embedding.reshape(1, hidden_dim, new_seq_length)
            new_pos_embedding = new_pos_embedding.permute(0, 2, 1)

            return new_pos_embedding
        
        return pos_embedding
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.last_hidden_state.mean(dim=1)
        x = self.norm(x)
        x = self.classifier(x)
        return x

class VJEPA2(nn.Module):
    def __init__(
        self, 
        model_name="MCG-NJU/videomae-large-finetuned-kinetics",
        new_image_size=64,
        num_classes=2
    ):
        super().__init__()
        
        # Load pretrained model
        self.model = VJEPA2Model.from_pretrained(model_name)
        # self.model = TimesformerModel.from_pretrained(model_name)
        config = self.model.config
        old_image_size = config.image_size
        config.image_size = new_image_size
        # Replace classifier
        hidden_size = self.model.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.last_hidden_state.mean(dim=1)
        x = self.norm(x)
        x = self.classifier(x)
        return x
