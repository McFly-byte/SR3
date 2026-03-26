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

    def __init__(self, query_dim, context_dim, n_heads=4, norm_groups=32):
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
        self.n_heads = n_heads
        # Zero-init output projection so the block starts as identity
        nn.init.zeros_(self.out_proj.weight)
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
        return x + self.out_proj(out)


class T1Encoder(nn.Module):
    """Lightweight encoder that extracts multi-scale structural features from the T1 image.

    The encoder produces a single feature map at the same spatial resolution as
    the input.  Cross-attention blocks inside UNet up-/down-sample as needed.
    Trained jointly with the diffusion model (no frozen weights).
    """

    def __init__(self, in_channels=1, out_channels=64, norm_groups=32):
        super().__init__()
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
        # Zero-init last layer so the encoder starts as a no-op
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
    ):
        super().__init__()

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
        if self.use_t1_cross_attn:
            t1_feat_dim = inner_channel
            self.t1_encoder = T1Encoder(
                in_channels=t1_in_channel,
                out_channels=t1_feat_dim,
                norm_groups=norm_groups,
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

    def forward(self, x, time, t1_feat=None):
        """
        x:       concatenated [condition, x_noisy] tensor
        time:    continuous sqrt_alpha_cumprod noise level (B, 1)
        t1_feat: optional pre-computed T1 encoder features (B, C_t1, H, W).
                 If None and self.use_t1_cross_attn is True, cross-attention
                 is skipped (backward-compatible).
        """
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

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
            else:
                x = layer(x)

        return self.final_conv(x)
