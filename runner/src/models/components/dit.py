import math
import typing
import warnings

import torchtune
import flash_attn

# from flash_attn.layers import rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.models.components import rotary

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def check_instance_UNetVDM(model: nn.Module) -> bool:
    from .vdm_unet import UNetVDM

    return isinstance(model, UNetVDM)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        #with torch.amp.autocast("cuda", enabled=False):
        #    cos, sin = rotary_cos_sin
        #    qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        if False:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seq_len,
                    step=seq_len,
                    dtype=torch.int32,
                    device=qkv.device,
                )
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0.0, causal=False
            )
            x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
        # else:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # qkv = rearrange(qkv, "(b s) ... -> b s ...", b=batch_size)
        # qkv = rearrange(qkv, "b s three h d -> b h three s d")
        q = rotary_cos_sin(qkv[:, :, 0])
        k = rotary_cos_sin(qkv[:, :, 1])
        v = qkv[:, :, 2]
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        #with sdpa_kernel(SDPBackend.MATH):
        #with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        #with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        x = scaled_dot_product_attention(q, k, v)
            #x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h s d -> b s (h d)")
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        self.vocab_dim = vocab_dim

    def forward(self, x):
        # check if x is onehot vector or not
        if x.ndim == 3 and x.shape[-1] == self.vocab_dim:
            result = torch.einsum("b s v, v d -> b s d", x, self.embedding)
            return result

        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma, output_hidden_states=False):
        x = self.vocab_embed(indices)
        x_init = x
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb

        logZ = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            logits = self.output_layer(x, c)

            if self.logZ_layer is not None:
                # logZ = self.logZ_layer(x, c).mean(dim=1)
                logZ = self.logZ_layer(x_init, c).mean(dim=1)

        out = [logits, logZ]
        if output_hidden_states:
            out.append([x])

        return out


class DITBetterHydra(DIT):
    def __init__(
        self,
        hidden_size,
        cond_dim,
        n_heads,
        dropout,
        scale_by_sigma,
        n_blocks,
        output_logZ,
        vocab_size: int,
    ):
        config = omegaconf.OmegaConf.create()
        config.model = omegaconf.OmegaConf.create()

        config.model.hidden_size = hidden_size
        config.model.cond_dim = cond_dim
        config.model.n_blocks = n_blocks
        config.model.n_heads = n_heads
        config.model.dropout = dropout
        config.model.scale_by_sigma = scale_by_sigma

        config.output_logZ = output_logZ

        super().__init__(config, vocab_size)


class WrappedPretrainedDIT(torch.nn.Module):
    def __init__(self, pretrained_backbone, output_logZ):
        super().__init__()

        self.pretrained_backbone = pretrained_backbone

        self.logZ_layer = None
        self.logZ_net = None
        if output_logZ:
            if isinstance(pretrained_backbone, DIT):
                vocab_size = pretrained_backbone.vocab_size
            elif check_instance_UNetVDM(pretrained_backbone):
                vocab_size = (
                    257  # hard-coded for celeba #pretrained_backbone.cfg.vocab_size
                )
            else:
                vocab_size = pretrained_backbone.config.vocab_size
            self.logZ_net = DITBetterHydra(
                hidden_size=192,
                cond_dim=64,
                n_blocks=2,
                n_heads=12,
                scale_by_sigma=True,
                dropout=0.1,
                output_logZ=True,
                vocab_size=vocab_size,
            )
        if False and output_logZ:
            hidden_dim, cond_dim = None, None
            if isinstance(pretrained_backbone, DIT):
                hidden_dim = pretrained_backbone.config.model.hidden_size
                cond_dim = pretrained_backbone.config.model.cond_dim
            elif check_instance_UNetVDM(pretrained_backbone):
                # hidden dim should be embedding dim * height * width
                # hack for now since currently for cifar n_blocks=32, for celeba n_blocks=64
                # height = pretrained_backbone.cfg.n_blocks
                hidden_dim = (
                    pretrained_backbone.cfg.embedding_dim * 4 * 4
                )  # * height * height  #pretrained_backbone.cfg.height * pretrained_backbone.cfg.width
                cond_dim = 4 * pretrained_backbone.cfg.embedding_dim
            else:
                hidden_dim = pretrained_backbone.backbone.config.hidden_dim
                cond_dim = pretrained_backbone.backbone.config.cond_dim

            self.logZ_layer = DDitFinalLayer(hidden_dim, 1, cond_dim)

        self.non_logZ_params_frozen = False

    def _get_backbone(self):
        if isinstance(self.pretrained_backbone, DIT):
            return self.pretrained_backbone
        elif check_instance_UNetVDM(self.pretrained_backbone):
            return self.pretrained_backbone
        else:
            return self.pretrained_backbone.backbone

    def forward(self, indices, sigma, output_logZ=True, output_logits=True):
        logits = None
        logZ = None

        if output_logits:
            outs = self.pretrained_backbone(indices, sigma, output_hidden_states=True)
            logits, hidden_states = None, None
            if len(outs) == 2:
                logits, hidden_states = outs
            else:
                logits, _, hidden_states = outs

        if output_logZ and self.logZ_net is not None:
            _, logZ = self.logZ_net(indices, sigma)
        if False and self.logZ_layer is not None:
            if isinstance(self.pretrained_backbone, DIT):
                cond = F.silu(self._get_backbone().sigma_map(sigma))
            elif check_instance_UNetVDM(self.pretrained_backbone):
                mean_dim = 0
                cond = self._get_backbone().sigma_map(sigma)
                hidden_states = hidden_states.detach()
            else:
                cond = F.silu(self._get_backbone().sigma_map(sigma))
                # warn that it may be unsupported
                warnings.warn("Using default sigma_map for WrappedPretrainedDIT")

            logZ = self.logZ_layer(hidden_states[-1].detach(), cond.detach()).mean(
                dim=1
            )

        return logits, logZ

    def parameters_only_logZ(self):
        if self.logZ_layer is not None:
            return self.logZ_layer.parameters()
        elif self.logZ_net is not None:
            return self.logZ_net.parameters()
        else:
            return []

    def parameters_no_logZ(self):
        return self.pretrained_backbone.parameters()

    def freeze_backbone_parameters(self):
        for parameter in self.pretrained_backbone.parameters():
            parameter.requires_grad = False

        self.non_logZ_params_frozen = True

    def unfreeze_backbone_parameters(self):
        for parameter in self.pretrained_backbone.parameters():
            parameter.requires_grad = True

        self.non_logZ_params_frozen = False


class DIT3D(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        hidden_size,
        cond_dim,
        n_heads,
        dropout,
        scale_by_sigma,
        n_blocks,
        n_particles,
        vocab_size: int,
    ):
        super().__init__()
        config = omegaconf.OmegaConf.create()
        config.model = omegaconf.OmegaConf.create()

        config.model.hidden_size = hidden_size
        config.model.cond_dim = cond_dim
        config.model.n_blocks = n_blocks
        config.model.n_heads = n_heads
        config.model.dropout = dropout
        config.model.scale_by_sigma = scale_by_sigma

        self.n_particles = n_particles
        self.vocab_size = vocab_size
        self.vocab_embed = torch.nn.Linear(vocab_size, config.model.hidden_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        #self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)
        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(
            dim=config.model.hidden_size // config.model.n_heads,
            base=10000,
            max_seq_len=1024
        )

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, t, x, *args, **kwargs):
        # eating args lie d_base
        t = t.squeeze()
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x.shape[0])
        x = x.reshape(-1, self.n_particles, self.vocab_size)
        x = self.vocab_embed(x)
        c = F.silu(self.sigma_map(t))
        rotary_cos_sin = self.rotary_emb
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            out = self.output_layer(x, c)
        out = out.reshape(x.shape[0], -1)
        return out
