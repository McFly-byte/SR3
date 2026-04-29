import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def _safe_group_count(channels, requested_groups=32):
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _init_module_scale(module, scale):
    if scale == 0:
        return _zero_module(module)
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=float(scale))
        else:
            nn.init.zeros_(p)
    return module

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups=32, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_safe_group_count(out_channels, norm_groups), out_channels),
            Swish(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionParser(nn.Module):
    """Split stacked MRSI condition channels using a configurable layout."""

    DEFAULT_LAYOUT = {
        "lr": [0, 1],
        "t1": [1, 2],
        "flair": [2, 3],
        "met_onehot": [3, 7],
        "mask": None,
    }

    def __init__(self, layout=None):
        super().__init__()
        self.layout = dict(self.DEFAULT_LAYOUT)
        if layout:
            self.layout.update(layout)

    def _slice(self, condition_x, key):
        spec = self.layout.get(key)
        if spec is None:
            return None
        if isinstance(spec, int):
            start, end = spec, spec + 1
        else:
            start, end = int(spec[0]), int(spec[1])
        if start < 0 or end > condition_x.shape[1] or end <= start:
            return None
        return condition_x[:, start:end, :, :]

    def forward(self, condition_x):
        if condition_x is None:
            return {}
        return {
            "lr": self._slice(condition_x, "lr"),
            "t1": self._slice(condition_x, "t1"),
            "flair": self._slice(condition_x, "flair"),
            "met_onehot": self._slice(condition_x, "met_onehot"),
            "mask": self._slice(condition_x, "mask"),
        }


class SobelEdges(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, x):
        edges = []
        for idx in range(x.shape[1]):
            xi = x[:, idx:idx + 1]
            gx = F.conv2d(xi, self.kernel_x.to(dtype=xi.dtype), padding=1)
            gy = F.conv2d(xi, self.kernel_y.to(dtype=xi.dtype), padding=1)
            edges.append(torch.sqrt(gx * gx + gy * gy + 1e-8))
        return torch.cat(edges, dim=1)


class MultiScaleConditionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=64, image_size=64, norm_groups=32):
        super().__init__()
        self.image_size = int(image_size)
        self.stem = nn.Sequential(
            ConvGNAct(in_channels, out_channels, norm_groups=norm_groups),
            ConvGNAct(out_channels, out_channels, norm_groups=norm_groups),
        )
        self.down32 = ConvGNAct(out_channels, out_channels, norm_groups=norm_groups, stride=2)
        self.down16 = ConvGNAct(out_channels, out_channels, norm_groups=norm_groups, stride=2)
        self.down8 = ConvGNAct(out_channels, out_channels, norm_groups=norm_groups, stride=2)

    def forward(self, x):
        feats = {}
        x = self.stem(x)
        feats[self.image_size] = x
        x = self.down32(x)
        feats[max(1, self.image_size // 2)] = x
        x = self.down16(x)
        feats[max(1, self.image_size // 4)] = x
        x = self.down8(x)
        feats[max(1, self.image_size // 8)] = x
        return feats


class StructureEncoder(nn.Module):
    def __init__(
        self,
        use_t1=True,
        use_flair=True,
        use_structure_edges=True,
        out_channels=64,
        image_size=64,
        norm_groups=32,
    ):
        super().__init__()
        self.use_t1 = bool(use_t1)
        self.use_flair = bool(use_flair)
        self.use_structure_edges = bool(use_structure_edges)
        base_channels = int(self.use_t1) + int(self.use_flair)
        if base_channels <= 0:
            base_channels = 1
        in_channels = base_channels * (2 if self.use_structure_edges else 1)
        self.edges = SobelEdges() if self.use_structure_edges else None
        self.encoder = MultiScaleConditionEncoder(
            in_channels, out_channels=out_channels, image_size=image_size, norm_groups=norm_groups
        )

    def forward(self, cond):
        xs = []
        if self.use_t1 and cond.get("t1") is not None:
            xs.append(cond["t1"])
        if self.use_flair and cond.get("flair") is not None:
            xs.append(cond["flair"])
        if not xs:
            return {}
        x = torch.cat(xs, dim=1)
        if self.edges is not None:
            x = torch.cat([x, self.edges(x)], dim=1)
        return self.encoder(x)


class LRMetabolicEncoder(nn.Module):
    def __init__(self, use_mask=False, out_channels=64, image_size=64, norm_groups=32):
        super().__init__()
        self.use_mask = bool(use_mask)
        in_channels = 1 + int(self.use_mask)
        self.encoder = MultiScaleConditionEncoder(
            in_channels, out_channels=out_channels, image_size=image_size, norm_groups=norm_groups
        )

    def forward(self, cond):
        lr = cond.get("lr")
        if lr is None:
            return {}
        xs = [lr]
        if self.use_mask and cond.get("mask") is not None:
            xs.append(cond["mask"])
        return self.encoder(torch.cat(xs, dim=1))


class MetaboliteEmbedding(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, embed_dim),
            Swish(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, met_onehot):
        if met_onehot is None:
            return None
        met_vec = met_onehot.mean(dim=(2, 3))
        return self.mlp(met_vec)


class GatedFusionBlock(nn.Module):
    def __init__(
        self,
        x_channels,
        adapter_channels=64,
        met_embed_dim=64,
        use_lr_adapter=True,
        use_structure_adapter=True,
        use_structure_gate=True,
        use_met_film=True,
        adapter_zero_init=True,
        adapter_init_scale=0.0,
        norm_groups=32,
    ):
        super().__init__()
        self.use_lr_adapter = bool(use_lr_adapter)
        self.use_structure_adapter = bool(use_structure_adapter)
        self.use_structure_gate = bool(use_structure_gate)
        self.use_met_film = bool(use_met_film)
        self.adapter_zero_init = bool(adapter_zero_init)
        self.adapter_init_scale = float(adapter_init_scale)

        if self.use_lr_adapter:
            self.lr_proj = nn.Conv2d(adapter_channels, x_channels, 1)
            self.alpha_lr = nn.Parameter(torch.tensor(1.0))
        else:
            self.lr_proj = None
            self.register_parameter("alpha_lr", None)

        if self.use_structure_adapter:
            self.struct_proj = nn.Conv2d(adapter_channels, x_channels, 1)
            self.alpha_struct = nn.Parameter(torch.tensor(1.0))
            if self.use_structure_gate:
                gate_in = x_channels + adapter_channels * (1 + int(self.use_lr_adapter))
                self.gate = nn.Sequential(
                    nn.Conv2d(gate_in, x_channels, 1),
                    nn.GroupNorm(_safe_group_count(x_channels, norm_groups), x_channels),
                    Swish(),
                    nn.Conv2d(x_channels, 1, 1),
                    nn.Sigmoid(),
                )
            else:
                self.gate = None
        else:
            self.struct_proj = None
            self.gate = None
            self.register_parameter("alpha_struct", None)

        self.met_film = nn.Linear(met_embed_dim, x_channels * 2) if self.use_met_film else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.lr_proj is not None:
            init_scale = self.adapter_init_scale if self.adapter_init_scale > 0 else 0.0
            _init_module_scale(self.lr_proj, init_scale if not self.adapter_zero_init or init_scale > 0 else 0.0)
        if self.struct_proj is not None:
            init_scale = self.adapter_init_scale if self.adapter_init_scale > 0 else 0.0
            _init_module_scale(self.struct_proj, init_scale if not self.adapter_zero_init or init_scale > 0 else 0.0)
        if self.met_film is not None:
            init_scale = self.adapter_init_scale if self.adapter_init_scale > 0 else 0.0
            _init_module_scale(self.met_film, init_scale if not self.adapter_zero_init or init_scale > 0 else 0.0)

    def forward(self, x, lr_feat=None, struct_feat=None, met_embed=None):
        if self.lr_proj is not None and lr_feat is not None:
            if lr_feat.shape[-2:] != x.shape[-2:]:
                lr_feat = F.interpolate(lr_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + self.alpha_lr * self.lr_proj(lr_feat)

        if self.struct_proj is not None and struct_feat is not None:
            if struct_feat.shape[-2:] != x.shape[-2:]:
                struct_feat = F.interpolate(struct_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            struct_delta = self.struct_proj(struct_feat)
            if self.gate is not None:
                gate_inputs = [x, struct_feat]
                if lr_feat is not None:
                    if lr_feat.shape[-2:] != x.shape[-2:]:
                        lr_feat = F.interpolate(lr_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    gate_inputs.append(lr_feat)
                elif self.use_lr_adapter:
                    gate_inputs.append(torch.zeros_like(struct_feat))
                struct_delta = struct_delta * self.gate(torch.cat(gate_inputs, dim=1))
            x = x + self.alpha_struct * struct_delta

        if self.met_film is not None and met_embed is not None:
            gamma, beta = self.met_film(met_embed).view(met_embed.shape[0], -1, 1, 1).chunk(2, dim=1)
            x = x * (1.0 + gamma) + beta
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class CrossAttentionBlock(nn.Module):
    """Spatial cross-attention that injects T1 structural context into decoder features.

    The query comes from the UNet decoder feature map; key/value come from the
    T1 encoder feature map (spatially aligned via bilinear interpolation when
    resolutions differ).  Uses a residual connection so the block is a no-op
    when the T1 context is absent or zero-initialized.
    """

    def __init__(self, query_dim, context_dim, n_heads=4, norm_groups=32, zero_init=True, init_scale=0.0):
        super().__init__()
        # Clamp norm_groups so it always divides query_dim
        norm_groups_q = min(norm_groups, query_dim)
        while query_dim % norm_groups_q != 0 and norm_groups_q > 1:
            norm_groups_q -= 1
        norm_groups_ctx = min(norm_groups, context_dim)
        while context_dim % norm_groups_ctx != 0 and norm_groups_ctx > 1:
            norm_groups_ctx -= 1

        self.norm_q = nn.GroupNorm(norm_groups_q, query_dim)
        self.norm_ctx = nn.GroupNorm(norm_groups_ctx, context_dim)
        # Project context to query space for K and V
        self.to_q = nn.Conv2d(query_dim, query_dim, 1, bias=False)
        self.to_k = nn.Conv2d(context_dim, query_dim, 1, bias=False)
        self.to_v = nn.Conv2d(context_dim, query_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(query_dim, query_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.n_heads = n_heads
        self.zero_init = bool(zero_init)
        self.init_scale = float(init_scale)
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            if self.init_scale > 0:
                nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.init_scale)
                nn.init.zeros_(self.out_proj.bias)
            else:
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)
        elif self.init_scale > 0:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.init_scale)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, context):
        """
        x:       (B, C_q, H, W)  decoder feature
        context: (B, C_ctx, H', W')  T1 encoder feature (any spatial size)
        """
        if context.shape[-2:] != x.shape[-2:]:
            context = F.interpolate(context, size=x.shape[-2:], mode='bilinear', align_corners=False)

        B, C, H, W = x.shape
        n_heads = self.n_heads
        head_dim = C // n_heads

        q = self.to_q(self.norm_q(x))          # (B, C, H, W)
        k = self.to_k(self.norm_ctx(context))  # (B, C, H, W)
        v = self.to_v(context)                 # (B, C, H, W)

        # Reshape to (B, heads, head_dim, H*W)
        q = q.view(B, n_heads, head_dim, H * W)
        k = k.view(B, n_heads, head_dim, H * W)
        v = v.view(B, n_heads, head_dim, H * W)

        # Attention: (B, heads, H*W, H*W)
        scale = math.sqrt(head_dim)
        attn = torch.einsum('bncd,bnce->bnde', q, k) / scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bnde,bnce->bncd', attn, v)  # (B, heads, head_dim, H*W)
        out = out.reshape(B, C, H, W)
        return x + self.alpha * self.out_proj(out)


class T1Encoder(nn.Module):
    """Lightweight encoder that extracts multi-scale structural features from the T1 image.

    The encoder produces a single feature map at the same spatial resolution as
    the input.  Cross-attention blocks inside UNet up-/down-sample as needed.
    Trained jointly with the diffusion model (no frozen weights).
    """

    def __init__(self, in_channels=1, out_channels=64, norm_groups=32, zero_init=True):
        super().__init__()
        self.zero_init = bool(zero_init)
        norm_groups_safe = min(norm_groups, out_channels)
        while out_channels % norm_groups_safe != 0 and norm_groups_safe > 1:
            norm_groups_safe -= 1

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(norm_groups_safe, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(norm_groups_safe, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, t1):
        return self.encoder(t1)


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        # T1 cross-attention: set t1_in_channel > 0 to enable
        t1_in_channel=0,
        t1_cross_attn_res=None,  # list of resolutions where cross-attn is applied in decoder
        t1_encoder_zero_init=False,
        adapter_zero_init=True,
        adapter_init_scale=1e-3,
        condition_layout=None,
        condition_adapter=None,
    ):
        super().__init__()
        condition_adapter = condition_adapter or {}
        self.condition_layout = condition_layout
        self.condition_adapter_cfg = condition_adapter
        self.condition_adapter_enabled = bool(condition_adapter.get("enabled", False))
        self.fusion_type = condition_adapter.get("fusion_type", "none")
        self.use_sgda_adapter = self.condition_adapter_enabled and self.fusion_type in ["sgda_gated_add", "sgda_film"]
        self.adapter_channels = int(condition_adapter.get("adapter_channels", inner_channel))
        self.use_met_film = bool(condition_adapter.get("use_met_film", False))
        self.adapter_zero_init = bool(condition_adapter.get("adapter_zero_init", adapter_zero_init))
        self.adapter_init_scale = float(condition_adapter.get("adapter_init_scale", adapter_init_scale))
        self.t1_encoder_zero_init = bool(t1_encoder_zero_init)

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        # Track decoder resolutions for cross-attention placement
        ups = []
        ups_res_track = []  # parallel list tracking resolution at each ups index
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                ups_res_track.append(now_res)
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                ups_res_track.append(now_res)
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        # T1 Cross-Attention setup
        self.use_t1_cross_attn = t1_in_channel > 0
        self._cross_attn_idx = {}  # always defined; populated below if enabled
        self.cross_attn_modules = nn.ModuleList()
        if self.use_t1_cross_attn:
            t1_feat_dim = inner_channel
            self.t1_encoder = T1Encoder(
                in_channels=t1_in_channel,
                out_channels=t1_feat_dim,
                norm_groups=norm_groups,
                zero_init=self.t1_encoder_zero_init,
            )
            # Determine which decoder positions get cross-attention
            if t1_cross_attn_res is None:
                # Default: apply at the same resolutions as self-attention
                t1_cross_attn_res = list(attn_res) if hasattr(attn_res, '__iter__') else [attn_res]
            t1_cross_attn_res_set = set(t1_cross_attn_res)

            cross_attn_modules = []
            for i, layer in enumerate(self.ups):
                if isinstance(layer, ResnetBlocWithAttn):
                    res_at_layer = ups_res_track[i]
                    if res_at_layer in t1_cross_attn_res_set:
                        # Derive output channel dim from the resnet block's second conv
                        out_ch = layer.res_block.block2.block[-1].out_channels
                        cross_attn_modules.append(
                            CrossAttentionBlock(
                                query_dim=out_ch,
                                context_dim=t1_feat_dim,
                                n_heads=max(1, out_ch // 64),
                                norm_groups=norm_groups,
                                zero_init=self.adapter_zero_init,
                                init_scale=self.adapter_init_scale,
                            )
                        )
                    else:
                        cross_attn_modules.append(None)
                else:
                    cross_attn_modules.append(None)

            # Store as ModuleList (skip None entries, keep index mapping)
            self.cross_attn_modules = nn.ModuleList(
                [m for m in cross_attn_modules if m is not None]
            )
            # Build index: ups_index -> cross_attn_modules index
            ca_idx = 0
            for i, m in enumerate(cross_attn_modules):
                if m is not None:
                    self._cross_attn_idx[i] = ca_idx
                    ca_idx += 1

        # SGDA-SR3 lightweight condition adapters. These are fully disabled for
        # old configs unless condition_adapter.enabled=true.
        self._fusion_idx = {}
        self.fusion_blocks = nn.ModuleList()
        if self.use_sgda_adapter:
            self.condition_parser = ConditionParser(condition_layout)
            fusion_res = condition_adapter.get("fusion_res", [16, 32])
            fusion_res_set = set(int(r) for r in fusion_res)
            self.lr_adapter = LRMetabolicEncoder(
                use_mask=condition_adapter.get("use_mask", False),
                out_channels=self.adapter_channels,
                image_size=image_size,
                norm_groups=norm_groups,
            ) if condition_adapter.get("use_lr_adapter", True) else None
            self.structure_adapter = StructureEncoder(
                use_t1=condition_adapter.get("use_t1", True),
                use_flair=condition_adapter.get("use_flair", True),
                use_structure_edges=condition_adapter.get("use_structure_edges", False),
                out_channels=self.adapter_channels,
                image_size=image_size,
                norm_groups=norm_groups,
            ) if condition_adapter.get("use_structure_adapter", True) else None
            self.met_embedding = MetaboliteEmbedding(
                in_channels=int(condition_adapter.get("met_onehot_channels", 4)),
                embed_dim=self.adapter_channels,
            ) if self.use_met_film else None

            fusion_blocks = []
            for i, layer in enumerate(self.ups):
                if isinstance(layer, ResnetBlocWithAttn) and int(ups_res_track[i]) in fusion_res_set:
                    out_ch = layer.res_block.block2.block[-1].out_channels
                    fusion_blocks.append(
                        GatedFusionBlock(
                            x_channels=out_ch,
                            adapter_channels=self.adapter_channels,
                            met_embed_dim=self.adapter_channels,
                            use_lr_adapter=condition_adapter.get("use_lr_adapter", True),
                            use_structure_adapter=condition_adapter.get("use_structure_adapter", True),
                            use_structure_gate=condition_adapter.get("structure_gate", True),
                            use_met_film=self.use_met_film,
                            adapter_zero_init=self.adapter_zero_init,
                            adapter_init_scale=self.adapter_init_scale,
                            norm_groups=norm_groups,
                        )
                    )
                else:
                    fusion_blocks.append(None)
            fb_idx = 0
            for i, block in enumerate(fusion_blocks):
                if block is not None:
                    self._fusion_idx[i] = fb_idx
                    self.fusion_blocks.append(block)
                    fb_idx += 1
        else:
            self.condition_parser = None
            self.lr_adapter = None
            self.structure_adapter = None
            self.met_embedding = None

    def reset_condition_branch_parameters(self):
        """Re-apply safe condition-branch initialization after global init_weights()."""
        if getattr(self, "use_t1_cross_attn", False):
            self.t1_encoder.reset_parameters()
            for module in self.cross_attn_modules:
                module.reset_parameters()
        if getattr(self, "use_sgda_adapter", False):
            for module in self.fusion_blocks:
                module.reset_parameters()

    def _prepare_sgda_condition(self, cond):
        if not self.use_sgda_adapter:
            return {}, {}, None
        parsed = cond or {}
        if not parsed and self.condition_parser is not None:
            return {}, {}, None
        lr_feats = self.lr_adapter(parsed) if self.lr_adapter is not None else {}
        struct_feats = self.structure_adapter(parsed) if self.structure_adapter is not None else {}
        met_embed = self.met_embedding(parsed.get("met_onehot")) if self.met_embedding is not None else None
        return lr_feats, struct_feats, met_embed

    def forward(self, x, time, t1_feat=None, cond=None):
        """
        x:       concatenated [condition, x_noisy] tensor
        time:    continuous sqrt_alpha_cumprod noise level (B, 1)
        t1_feat: optional pre-computed T1 encoder features (B, C_t1, H, W).
                 If None and self.use_t1_cross_attn is True, cross-attention
                 is skipped (backward-compatible).
        cond:    optional parsed condition dict for SGDA adapters.
        """
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        lr_feats, struct_feats, met_embed = self._prepare_sgda_condition(cond)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                # Apply T1 cross-attention if available at this layer
                if t1_feat is not None and i in self._cross_attn_idx:
                    ca_mod = self.cross_attn_modules[self._cross_attn_idx[i]]
                    x = ca_mod(x, t1_feat)
                if i in self._fusion_idx:
                    res = x.shape[-1]
                    fusion = self.fusion_blocks[self._fusion_idx[i]]
                    x = fusion(
                        x,
                        lr_feat=lr_feats.get(res),
                        struct_feat=struct_feats.get(res),
                        met_embed=met_embed,
                    )
            else:
                x = layer(x)

        return self.final_conv(x)
