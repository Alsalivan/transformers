# coding=utf-8
# Copyright 2022 Multimedia Computing Group, Nanjing University and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VideoMAE (masked autoencoder) model."""


import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig
from ...models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEForPreTrainingOutput

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VideoMAEConfig"
_CHECKPOINT_FOR_DOC = "MCG-NJU/videomae-base"


@dataclass
class VideoMAEDecoderOutput(ModelOutput):
    """
    Class for VideoMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VideoMAEForPreTrainingOutput(ModelOutput):
    """
    Class for VideoMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob (float): Probability of dropping a path. Default: 0.0
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying DropPath.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndimension() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output   

class LayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layerscale_c = nn.Parameter(config.layerscale_init_values * torch.ones(config.hidden_size))

    def forward(self, x):
        return x * self.layerscale_c
    
def get_multi_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Parameters:
    - embed_dim: int, the output dimension for each position.
    - grid_size: list[int], dimensions of the grid, e.g., [8, 12, 12] for a 3D grid.
    - add_cls_token: bool, if True, adds a zero embedding vector for a class token at the beginning.
    
    Returns:
    - torch.Tensor, positional embeddings with shape [1, np.prod(grid_size) + (1 if add_cls_token else 0), embed_dim].
    """
    grid_dim = len(grid_size)
    assert grid_dim >= 2, "Grid_size should be at least 2D"
    assert embed_dim % (grid_dim * 2) == 0, "Each dimension has 2 channels (sin, cos)"

    # Creates a grid of coordinates (e.g., 3D coordinates for each point in an 8x12x12 grid).
    grid = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in grid_size], indexing='ij')
    grid = torch.stack(grid, dim=0)  # Stacks to create a single tensor representing all coordinates.

    pos_embed = get_multi_sincos_pos_embed_from_grid(embed_dim, grid)

    if add_cls_token:
        pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    
    return pos_embed.unsqueeze(0)

def get_multi_sincos_pos_embed_from_grid(embed_dim, grid):
    grid_dim = len(grid.shape) - 1  # Number of grid dimensions.
    # Generate embeddings for each dimension and concatenate them.
    emb = [get_1d_sincos_pos_embed_from_grid(embed_dim // grid_dim, grid[i]) for i in range(grid.shape[0])]
    emb = torch.concatenate(emb, dim=1) # (T*H*W, D/4) -> (T*H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
    
class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """
    def __init__(self, config):
        super().__init__()
        num_channels = config.num_channels
        hidden_size = config.hidden_size

        self.patch_size = config.patch_size
        self.tubelet_size = int(config.tubelet_size)

        self.num_channels = num_channels

        self.grid_size = (config.num_frames // config.tubelet_size,
                          config.image_size // config.patch_size,
                          config.image_size // config.patch_size)

        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size),
        )

    def forward(self, pixel_values):
        # (batch_size, num_channels, num_frames, height, width)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

class VideoMAEEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = VideoMAEPatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_cls_token else None

        self.grid_size = self.patch_embeddings.grid_size
        self.patch_size = self.patch_embeddings.patch_size
        self.tubelet_size = self.patch_embeddings.tubelet_size

        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        if config.use_cls_token:
            num_patches += 1
        
        if config.use_learnable_pos_emb:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=True)
        else:
            self.position_embeddings = nn.Parameter(
                get_multi_sincos_pos_embed(
                    grid_size=(self.grid_size[0], self.grid_size[1], self.grid_size[2]),
                    embed_dim=config.hidden_size,
                    add_cls_token=config.use_cls_token
                    ),
                requires_grad=False
                )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
    
    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio) + 0.5)

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore

    def forward(self, pixel_values, noise=None, apply_masking=True):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        if self.config.use_learnable_pos_emb:
            position_embeddings = self.position_embeddings.type_as(embeddings).to(embeddings.device)
        else:
            position_embeddings = self.position_embeddings.type_as(embeddings).to(embeddings.device).detach() # (b, num_patches+1/num_patches, embed_dim)
        
        # Add position embeddings without cls token
        if self.config.use_cls_token:
            embeddings = embeddings + position_embeddings[:, 1:, :]
        else:
            embeddings = embeddings + position_embeddings

        # Masking: length -> length * config.mask_ratio
        if apply_masking:
            embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        else:
            mask, ids_restore = None, None

        if self.config.use_cls_token:
            # Append the CLS token to the sequence
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cls_tokens = cls_tokens + position_embeddings[:, :1, :]
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings, mask, ids_restore


class CosAttention(nn.Module):
    def __init__(self, config: VideoMAEConfig, qk_scale=None) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.scale = nn.Parameter(torch.log(10 * torch.ones((self.num_attention_heads, 1, 1)))) if qk_scale is None else qk_scale
        self.qkv = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=True)

        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, x, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        if head_mask is not None:
            # Apply head mask after computing the attention scores
            attn = attn * head_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        if output_attentions:
            return (x, attn)
        return (x, )
    

class VideoMAESelfAttention(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)

        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values)
        query_layer = self.transpose_for_scores(queries)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->VideoMAE
class VideoMAESelfOutput(nn.Module):
    """
    The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(input_tensor)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->VideoMAE
class VideoMAEAttention(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        if config.attention_type == 'cosine':
            self.attention = CosAttention(config)
        elif config.attention_type == 'self_attention':
            self.attention = VideoMAESelfAttention(config)
        self.output = VideoMAESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->VideoMAE
class VideoMAEIntermediate(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # ommit this for the orignal BERT implement

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->VideoMAE
class VideoMAEOutput(nn.Module):
    def __init__(self, config: VideoMAEConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layerscale = LayerScale(config=config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.drop_path(self.layerscale(hidden_states)) + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->VideoMAE
class VideoMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: VideoMAEConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VideoMAEAttention(config)
        
        self.intermediate = VideoMAEIntermediate(config)
        self.output = VideoMAEOutput(config, drop_path_rate=drop_path_rate)

        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=torch.float32)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=torch.float32)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layerscale = LayerScale(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in VideoMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(self.layerscale(attention_output)) + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VideoMAE
class VideoMAEEncoder(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.config = config
        
        depth = config.num_hidden_layers
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.layer = nn.ModuleList([VideoMAELayer(config, drop_path_rate=dpr[i]) for i in range(depth)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class VideoMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VideoMAEConfig
    base_model_prefix = "videomae"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)

VIDEOMAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VideoMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIDEOMAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.__call__`] for details.
        
        apply_masking (bool`):
            Apply random patch masking or not

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEModel(VideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VideoMAEEmbeddings(config)
        self.encoder = VideoMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=torch.float32)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        apply_masking: bool = True, 
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_layernorm: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
            batch must have the same number of masked patches. If `None`, then all patches are considered. Sequence
            length is `(num_frames // tubelet_size) * (image_size // patch_size) ** 2`.

        Returns:

        Examples:

        ```python
        >>> import av
        >>> import numpy as np

        >>> from transformers import AutoImageProcessor, VideoMAEModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 16 frames
        >>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        >>> # prepare video for the model
        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1568, 768]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, apply_masking=apply_masking)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        
        if use_layernorm:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VideoMAEDecoder(nn.Module):
    def __init__(self, config, grid_size, patch_size, tubelet_size):
        super().__init__()

        self.grid_size = grid_size # (num_t_patches, num_x_patches, num_y_patches)

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size), requires_grad=True)

        num_patches = grid_size[0] * grid_size[1] * grid_size[2]

        if config.use_cls_token:
            num_patches += 1

        if config.use_learnable_pos_emb:
            self.decoder_position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches, config.decoder_hidden_size),
                requires_grad=True
            )
        else:
            self.decoder_position_embeddings = nn.Parameter(
                get_multi_sincos_pos_embed(
                    grid_size=(self.grid_size[0], self.grid_size[1], self.grid_size[2]),
                    embed_dim=config.decoder_hidden_size,
                    add_cls_token=config.use_cls_token
                ),
                requires_grad=False
            )

        depth = decoder_config.num_hidden_layers
        
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_layers = nn.ModuleList([VideoMAELayer(decoder_config, drop_path_rate=dpr[i]) for i in range(depth)])

        self.layernorm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps, dtype=torch.float32)
        self.head = nn.Linear(config.decoder_hidden_size, patch_size*patch_size*tubelet_size, bias=True)

        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):  
        batch_size, num_visible, hidden_dim = hidden_states.shape
        original_length = ids_restore.shape[1]
        
        if self.config.use_cls_token:
            num_visible -= 1

        num_mask_tokens = original_length - num_visible

        mask_tokens = self.mask_token.repeat(batch_size, num_mask_tokens, 1)

        if self.config.use_cls_token:
            hidden_states_ = torch.cat([hidden_states[:, 1:, :], mask_tokens], dim=1)  # no cls token
        else:
            hidden_states_ = torch.cat([hidden_states, mask_tokens], dim=1)

        # Unshuffle to restore the original order of tokens including the newly added mask tokens
        hidden_states_ = torch.gather(hidden_states_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_dim))

        if self.config.use_cls_token:
            x = torch.cat([hidden_states[:, :1, :], hidden_states_], dim=1)  # append cls token
        else:
            x = hidden_states_

        if self.config.use_learnable_pos_emb:
            position_embeddings = self.decoder_position_embeddings.type_as(x).to(x.device)
        else:
            position_embeddings = self.decoder_position_embeddings.type_as(x).to(x.device).detach() # (b, num_patches+1/num_patches, embed_dim)
        
        hidden_states = x + position_embeddings

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # predictor projection
        hidden_states = self.layernorm(hidden_states)
        logits = self.head(hidden_states)

        if self.config.use_cls_token:
            logits = logits[:, 1:, :]  # remove cls token

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return VideoMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
            )


@add_start_docstrings(
    "The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForPreTraining(VideoMAEPreTrainedModel):
    config_class = VideoMAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.videomae = VideoMAEModel(config)

        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

        self.patch_size = self.videomae.embeddings.patch_size
        self.tubelet_size = self.videomae.embeddings.tubelet_size
        self.grid_size = self.videomae.embeddings.grid_size

        self.decoder = VideoMAEDecoder(config, grid_size=self.grid_size, patch_size=self.patch_size, tubelet_size=self.tubelet_size)

        # Initialize weights and apply final processing
        self.post_init()

    def patchify(self, pixel_values):
        """
        Convert pixel values to patches.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, depth, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patches of pixel values.
        """
        batch_size, num_channels, depth, height, width = pixel_values.shape

        # Split depth into tubelets
        pixel_values = pixel_values.unfold(2, self.tubelet_size, self.tubelet_size)
        depth = depth // self.tubelet_size

        # Split height and width into patches
        pixel_values = pixel_values.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)

        # Reshape to (batch_size, num_patches, patch_size*patch_size*tubelet_size*num_channels)
        patches = pixel_values.contiguous().view(batch_size, num_channels, depth, -1, self.patch_size * self.patch_size * self.tubelet_size)

        # Merge depth and patch dimensions
        patches = patches.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, self.patch_size * self.patch_size * self.tubelet_size * num_channels)

        return patches

    def unpatchify(self, patchified_pixel_values):
        """
        Convert patches back to original pixel values.

        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, tubelet_size * patch_height * patch_width * num_channels)`):
                Patchified 3D pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, depth, height, width)`:
                Unpatchified 3D pixel values.
        """
        batch_size, num_patches, patch_dim = patchified_pixel_values.shape

        # Calculate the original dimensions
        num_channels = patch_dim // (self.tubelet_size * self.patch_size * self.patch_size)
        depth = self.grid_size[0] * self.tubelet_size
        height = self.grid_size[1] * self.patch_size
        width = self.grid_size[2] * self.patch_size

        # Reshape patches to the intermediate form
        patches = patchified_pixel_values.view(batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2], num_channels, self.tubelet_size, self.patch_size, self.patch_size)
        
        # Permute and reshape to restore the original shape
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(batch_size, num_channels, depth, height, width)

        return patches

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, depth, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss
        """
        target = self.patchify(pixel_values)

        # Compute reconstruction loss
        loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
        loss = loss.mean(dim=-1)

        if self.config.mask_loss:
            reconstruction_loss = (loss * mask).sum() / mask.sum().item()  # mean loss on removed patches 
        else:
            reconstruction_loss = loss.mean()

        return reconstruction_loss

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        apply_masking: bool = True,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_layernorm: bool = True,
    ) -> Union[tuple, VideoMAEForPreTrainingOutput]:
        r"""
        apply_masking (`bool`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
            batch must have the same number of masked patches. Sequence length is `(num_frames // tubelet_size) *
            (image_size // patch_size) ** 2`.
        

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, VideoMAEForPreTraining
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 16
        >>> video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

        >>> pixel_values = image_processor(video, return_tensors="pt").pixel_values

        >>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
        >>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
        >>> bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.videomae(
            pixel_values,
            apply_masking=apply_masking,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_layernorm=use_layernorm,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        sequence_output = self.encoder_to_decoder(latent)

        decoder_outputs = self.decoder(sequence_output, ids_restore)
        logits = decoder_outputs.logits

        loss = self.forward_loss(pixel_values=pixel_values, pred=logits, mask=mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )