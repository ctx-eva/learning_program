# ViT

Paper: [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf)

<image src="./images/ViT.jpeg">

ViT通过将输入图像转化为一系列小块,作为序列输入Transformer中,利用Transformer的自注意力机制捕捉图像中的长距离依赖关系,改善了原先卷积网络感受野范围有限.
并证明了Transformer架构不仅适用于文本数据同样适用于图像数据。ViT是一种Transformer Encoder网络。

<details>
<summary>code for ViT</summary>

```python
import torch
from torch import nn

# 引入einpos包,方便处理复杂的维度重排列
from einops import rearrange, repeat 
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 通过使用einpos改写transformer自注意力
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1) # chunk()会自动补齐某种类似split()操作
        # 和view(b, n, h, d).transpose(1, 2)效果一样
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # dots: [b h n n]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # out:  [b h n d] matmul的操作参考torch.matmul中的第五点
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # 同过einops.Rearrange可以轻松实现patch的划分，将image_height拆成h个patch_height，width亦然
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        ) # out : [b (h w) dim]

        # learnable pos_embed,num_patches个image patch + 1个cls_token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # ViT在所有输入序列前添加cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # transformer输出接多层全连接做分类
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 拼接cls token和图像patch输入
        x = torch.cat((cls_tokens, x), dim=1)
        # 两个维度大于1的Tensor计算操作会自动补齐维度(broadcastable)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # pool不为'mean'时取cls_token的输出
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
```
</details>

# Swin-Transformer

Paper: [Swin transformer: hierarchical vision transformer using shifted windows](https://arxiv.org/pdf/2103.14030.pdf)

<image src="./images/Swintransformer_architecture.png">

<image src="./images/Swintransformer_winshift.jpg">

Swin-Transformer来解决ViT由于采用16x16的patch,使得feature map只包含大尺度上的特征.通过生成多尺度的feature map,Swin-Transformer可以迁移到使用特征金字塔结构的的网络结构中。Swin-Trasformer通过将patch按照对应原图的位置分割成多个window,计算window内的attn(W-MSA)实现计算量的缩小(相比于ViT采用16x16的patch,Swin-Transformer采用2x2的patch,保留更多的特征但,patch数量大大增加)。由于WSA只包含local attn,通过引入window shift来实现长距离上的attn,对SW-MSA的attn结果只保留原本就连续的patch对应位置的值。

The code comes from https://github.com/microsoft/Swin-Transformer/

<details>
<summary>code for Swin-transformer</summary>

```python
# comes from /models/swin_transformer.py
from einpos import rearange, repeat

def window_partition(x, window_size):
    """
    将patch拆分成多个window
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    # 参考Twins-SVT 引入einops 以上可以写成
    # windows = rearange(x, 'b (y w1) (x w2) c -> (b y x) w1 w2 c', w1 = window_size, w2 = window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将feature map从window block还原到原始状态
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    # x = rearange(windows, '(b y x) w1 w2 c -> b (y w1) (x w2) c', y = H//window_size, x = W//window_size)
    return x

class WindowAttention(nn.Module):
    # WindowAttention在最基本的Transformer MSA block中附加了表示window内patch相对位置关系的relative_position_bias_table
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        #relative_position_index表示在一个window内两个patch之间的相对关系
        # 这里为啥不直接用window内xy方向的的两个learable的向量合成代替。
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # 通过relative_position_index在learnable的relative_position_bias_table中选取，表示每个window中patch的相对位置，
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # 当SW-MSA时需要计算softmax的mask,将shift后在原feature map中不相邻的patch间的attn值置0
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            # 确定进行移位后不相邻且在一个window内的需要做attn的patch,对shift前且在同一个window内的patch赋相同的值
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # 该过程应该可以由以下代码完成
            # img_mask[:,-self.shift_size:,:,:] += h_shift_v
            # img_mask[:,:,-self.shift_size:,:] += w_shift_v

            # 将img_mask转化到各个window内
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 通过相减判断进行pot-product的patch是否来自同一window
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # attn_mask附加在softmax前，对相应位置减去一个较大的值，通过softmax后该位置值趋于0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                # 使用torch.roll在feature map的HW方向上进行循环移位,实现window shift.也因为torch.roll在其他框架内未实现所以有Twins-SVT等改进
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows 对feature按window拆分
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        # 在完成attn layer后，通过torch.roll和window_reverse将attn结果转换为原始的feature map形式
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    # 每个stage后通过patchmerge缩小feature map尺寸
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 这个过程可以用以下代替
        # x = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    # 如图所示基本上每个layer包含一个W-MSA和一个SW-MSA模块
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
```

</details>

# Twins-SVT

Paper: [Twins: revisiting the design of spatial attention in vision transformers](https://arxiv.org/pdf/2104.13840.pdf)

<image src="./images/Twins-PCPVT-S.png">

由于attn vector做dot-product attn运算时计算量和vector的分辨率的二次方成正相关,Twins-SVT通过缩减vector的分辨率增加通道数,降低dot-product的运算量,
使其和vector的分辨率的一次成正相关。该方法类似于深度可分离卷积。在文中被称为spatially separable self-attention(SSSA).
Twins-SVT通过引入Global sub-sampled attention (GSA)模块解决Swin-Transformer进行窗口滑动导致对于不同框架的部署困难问题

<details>
<summary>code for Twins-SVT</summary>

```python
# The layer block in Twins-SVT
layers.append(nn.Sequential(
    PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
    Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
    PEG(dim = dim_next, kernel_size = peg_kernel_size),
    Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
))

# PatchEmbedding block 
class PatchEmbedding(nn.Module):
    '''
    PatchEmbedding和在ViT中to_patch_embedding操作一样，将输入打包成patch^2的块，并进行Embed
    和ViT不同的是在Twins-SVT中，对每个layer stage都进行一次，随着stage depth的加深，feature map分辨率缩小
    '''
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            LayerNorm(patch_size ** 2 * dim),
            nn.Conv2d(patch_size ** 2 * dim, dim_out, 1),
            LayerNorm(dim_out)
        )

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        # shape : [b c H W] -> [b (c p1 p2) H/p1 W/p2]
        return self.proj(fmap) # shape : [b (c p1 p2) h w] -> [b dim_out h w]

# Transformer block详情
for _ in range(depth):
    self.layers.append(nn.ModuleList([
        Residual(LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = local_patch_size)) if has_local else nn.Identity(),
        Residual(FeedForward(dim, mlp_mult, dropout = dropout)) if has_local else nn.Identity(),
        Residual(GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = global_k)),
        Residual(FeedForward(dim, mlp_mult, dropout = dropout))
    ]))

# Loclly-ground self-attention (LSA) block
class LocalAttention(nn.Module):
    '''
    LSA block 可以认为是对PatchEmbedding后的feature做patch_size的分块序列，经过一个transformer的Muitl head self-attn
    和ViT处理输入以及其后的第一个self-attn模块一致。
    经过LSA block的输入输出尺寸不发生变化
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        # input: [b d_in H W]
        fmap = self.norm(fmap)
        # 因为经过PatchEmbedding block后每个模块的dim = d_in，fmap: [h d_in H W]

        shape, p = fmap.shape, self.patch_size
        # 在LSA(local self-attention)中x,y表示x*y个patch
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))
        # 用map做了function (x//p,y//p)计算拆分成多少个sub-window
        
        # 这里的x,y表示有x*y个sub-window,每个window包含p1*p2个patch
        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)
        # fmap : [(b H/p W/p) d_in p p]

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        # q,k,v :  [(b H/p W/p) inner_dim p p]
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), (q, k, v))
        # q,k,v :  [(b H/p W/p heads) (p p) dim_head]

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # dots: [(b H/p W/p heads) (p p) (p p)]
        # dot-product的计算量和q,k分辨率的二次方相关,转变维度后计算量缩小

        attn = dots.softmax(dim = - 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        # out : [(b H/p W/p heads) (p p) dim_head]
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h = h, x = x, y = y, p1 = p, p2 = p)
        # out : [b inner_dim H W]
        return self.to_out(out) # out : [b d_in H W]

# Global sub-sampled attention (GSA) block
class GlobalAttention(nn.Module):
    '''
    GSA block 通过在dot-product attentio中k,v使用空洞卷积，扩大感受野的同时缩减dot-product attention的计算量
    经过GSA block的输入输出尺寸不发生变化
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride = k, bias = False)
        # GSA利用空洞卷积扩大感受野的同时缩减dot-product attention的计算量

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        # q : [b inner_dim H W], k,v : [b inner_dim H/k W/k]

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))
        # q : [(b heads) (H W) dim_head], k,v : [(b heads) (H/k W/k) dim_head]

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q : [(b heads) (H W) (H/k W/k)]

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        # out : [(b heads) (H W) dim_head]

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        # out : [b inner_dim H W]
        return self.to_out(out) # out : [b dim_in H W]

# 继承自CPVT的position encoding generator(PEG), 
# CPVT证明通过zero pad使用DepthWise Conv可以通过feature map感知patch的absolute position information
# CPVT显示将PEG layer放在第一个encoder layer后对性能可以起到最大的效果
class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        # Depthwise conv给每个通道分配一个卷积核，通道之间并不发生数据交流。在Pytorch中可以通过分组卷积实现
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1))

    def forward(self, x):
        return self.proj(x)
```
</details>

# NaViT

Paper: [Patch n'pack: navit, a vision transformer for any aspect ratio and resolution](https://arxiv.org/pdf/2307.06304.pdf)

NaViT通过序列打包（Patch n’ Pack）的方式，实现了对任意分辨率和宽高比的输入进行处理，提升模型在多种分辨率下的性能，并且增大了模型吞吐量提升了训练效率。

<details>
<summary>code for NaViT</summary>

```python
def group_images_by_max_seq_len(images: List[Tensor], patch_size: int, calc_token_dropout = None,
        max_seq_len = 2048) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)
        # 当前图片包含的patch个数
        image_seq_len = (ph * pw)
        # 这里直接可以等于int(image_seq_len*(1-calc_token_dropout)),表示单张图片drop后的token长度
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        # 单张图片drop后的token长度要比模型允许的输入序列的长度短
        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            # 当前group聚合序列的长度seq_len大于模型允许的输入序列的长度,转入下一个序列
            groups.append(group)
            group = []
            seq_len = 0

        # group表示当前单个batch的序列包含的图片
        group.append(image)
        # 单个group聚合序列的长度seq_len
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)
    # groups内每个元素包含一个batch内包含的图片
    return groups

class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., 
        emb_dropout = 0., token_dropout_prob = None):
        super().__init__()
        image_height, image_width = pair(image_size) # 函数pair()用于判断image_size的数据类型并将其分配给height,width

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None
        # 解释来源Kimi AI,callable判断对象是否可调用，即对象是否是函数或者实现了__call__()方法
        if callable(token_dropout_prob): 
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            # 解释来源Kimi AI,该句定义了self.calc_token_dropout函数，接收两个变量，返回token_dropout_prob的值
            self.calc_token_dropout = lambda height, width: token_dropout_prob
        # 以上这些过程只是实现了calc_token_dropout的设置，感觉直接设置为float变量在后面调用时并不影响

        # calculate patching related stuff
        # 如红字所示确保输入网络的图像长宽可以被patch_size整除
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2) #每张图片输入总patch维度

        self.channels = channels
        self.patch_size = patch_size

        # 在这里是对drop和concatenate处理完后的patch序列
        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        # 因为NaVit通过drop和concatenate解构了图像形状,需要hpos和wpos,这里是learnable的形式
        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries
        # 因为NaVit在同一个batch中包含不同张数的图像,不能和一般的ViT那样添加cls token
        # 按照patch所属图片整理完Encoder的output后，添加一个learnable的cls vector,
        # 利用transformer Decoder结构,训练分类
        # 参考来源《Going deeper with image transformers》Section 3
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], # assume different resolution images already grouped correctly
        group_images = False,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)

        arange = partial(torch.arange, device = device)
        # orig_pad_sequence:torch.nn.utils.rnn.pad_sequence
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        # auto pack if specified
        if group_images:
            # 根据图片包含的patch长度和模型的输入序列长度max_seq_len,确定每个group即每个batch内包含哪几张图
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )

        # process images into variable lengthed sequences with attention mask
        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            # 取出一个group中包含多张图片的batch,num_images记录该batch中有几张图
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)

                # meshgrid组合0~ph,0~pw表示当前图像每个patch的h,w位置
                # pos:[h,w,2]
                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)
                # 将pos转换为向量
                pos = rearrange(pos, 'h w c -> (h w) c')
                # 将image元素转变为长度为h_patch*w_patch的向量
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)
                # 长度为h_patch*w_patch，一共有多少个patch
                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    # 计算drop后的序列长度
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    # 随机保留,使用torch.randn生成序列长度的随机数,使用tensor.topk取indices计算保留元素的序号
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices
                    # 获取drop后的image的patch元素和该patch的pos的h,w绝对位置
                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                # 在原image_ids的右侧扩展长度为当前seq的长度(drop后)值为当前图像序号image_id
                # image_ids表示该batch中每个元素来源的序号
                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                # sequences聚合一个batch中的image的patch元素,positions聚合一个batch中的image的patch位置
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            # torch.cat聚合成向量,batch_中存放所有batch的数据
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        # derive key padding mask
        # length记录每个batch中的序列长度
        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        # max_length值[0~maxlength-1]
        max_length = arange(lengths.amax().item())
        # key_pad_mask:[b maxlength],每个batch中小于该batch序列的位置为True
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        # derive attention mask, and combine with key padding mask from above
        # torch.nn.utils.rnn.pad_sequence,因为batch_first = True,tensor从batch个[L *]->[B maxlength *]
        batched_image_ids = pad_sequence(batched_image_ids)
        # self attn mask,元素属于同一图片,attn_mask:[b 1 maxlength maxlength]
        # 这里的attn mask形状似某种呈对角线分布的正方形块
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        # mask大于每个batch的length的元素置false
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')

        # combine patched images as well as the patched width / height positions for 2d positional embedding
        # patches:[b maxlength (c p1 p2)],patch_positions:[b maxlength 2]
        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)        

        # to patches
        # input embedding x:[b maxlength (c p1 p2)] -> [b maxlength dim]
        x = self.to_patch_embedding(patches)        

        # factorized 2d absolute positional embedding
        # 按最后一维拆解，patch_positions分成h和w方向的绝对距离,h_indices:[b maxlength],w_indices:[b maxlength]
        h_indices, w_indices = patch_positions.unbind(dim = -1)
        # 按照h_indices表示的索引从self.pos_embed_height取值构成h_pos,w_pos也同样
        # 这里用的是learned positional embedding不是fixed positional embedding。保留了图像的长宽比
        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]
        # x:transformer input seq
        x = x + h_pos + w_pos

        # embed dropout
        x = self.dropout(x)

        # attention
        # simple multi head attn transformer encoder layer
        # The encoder with LayerNorm bias=0, and use RMSNorm in q,k
        x = self.transformer(x, attn_mask = attn_mask)
        # x out:[b maxlength dim]

        # do attention pooling at the end
        # attention pooling 在这里主要用于实现每张图片的类别计算
        max_queries = num_images.amax().item() # 一个batch中最多包含几张图
        # attention pooling的learnable cls token,queries:[b max_queries dim]
        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        # attention pool mask

        image_id_arange = arange(max_queries)
        # attn_pool_mask:[b max_queries maxlength],cross attn mask
        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
        # 掩去对每个batch的length和maxlength之间padding的元素
        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')
        # attn_pool_mask:[b 1 max_queries maxlength]
        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

        # attention pool，Do cross attn between cls_token and transformer encoder output
        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries
        # x out:[b max_queries dim]
        # x: [b*max_queries dim], 每个batch中每个图片表示成dim长度的向量
        x = rearrange(x, 'b n d -> (b n) d')

        # each batch element may not have same amount of images
        # is_images:[b max_queries],表示该位置是否有输入图片
        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        is_images = rearrange(is_images, 'b n -> (b n)')
        # x:[all_image_lens dim]
        x = x[is_images]

        # project out to logits

        x = self.to_latent(x)
        # 函数输出 [all_image_lens num_class],做分类训练
        return self.mlp_head(x)
```

LayerNorm实现对方程 $y=\frac{x-E(x)}{\sqrt{Var[x]+\epsilon}}*\gamma$ 的计算
```python
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        # 通过使用torch.nn.Module.register_buffer,将layerNorm的bias排除出模型结构,恒为0，且不更新
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        # torch.nn.functional.layer_norm实现方程计算
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
```

RMSNorm实现对方程 $y=\frac{x}{\sqrt{\frac{1}{d}\sum^{d}_{i=1}x^2_i}}*\gamma$ 的计算,通过不计算均值和方差缩减计算量
```python
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        # torch.nn.functional.normalize实现L2 normalization
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma
```
</details>

# MAE

Paper: [Masked autoencoders are scalable vision learners](https://arxiv.org/pdf/2111.06377.pdf)

<image src="./images/MAE.png" style="width: 50%; height: auto;">

MAE(Masked AutoEncoder)通过随机掩码输入图像的一部分，然后训练模型重建被掩码覆盖的区域，使得模型能够学习到图像的语义特征。

<details>
<summary>code for MAE</summary>

```python
class MAE(nn.Module):
    def __init__(self, *, encoder, decoder_dim, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64):
        # 这里encoder代表pretrain的模型
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        # 以vit_pytorch中vit为例to_patch是对图像rerange组成patch，patch_to_emb即input embed
        self.to_patch = encoder.to_patch_embedding[0] 
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        # pixel_values_per_patch=input.channel*p1*p2,一个patch中包含多少个像素值
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # 构建解码器在训练中用以恢复原图像的像素值
        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        # 为啥这里的decoder pos emb对patch_id用nn.Embedding,而不是和一般tr那样使用和input一样类型的emb
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        # 全连接将维度变换成一个patch包含的像素值
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device
        # get patches
        patches = self.to_patch(img) 
        batch, num_patches, *_ = patches.shape
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # 对ViT这类在input token中添加CLS token需要排除第一个cls token的embed,在MAE的过程中用不到该token
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # 对Twins-SVT这类使用GAP或者Attn pool不包含CLS token的直接使用其pos emb
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        # 计算需要mask的patch序号,通过rand后argsort,按需要的长度截取
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        #[batch_range, unmasked_indices] unmask patch的序号
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        # unmasked token输入要pretrained模型
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 使用masked token和pretrained output token组合起来
        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        # 按照mask和unmask序号选择token对decoder_tokens进行赋值
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        # 将tr模型Decoder的output的维度变换到原图像的像素值个
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        # 计算mse loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss
```
</details>

# Dino

Paper: [Emerging properties in self-supervised vision transfoemers](https://arxiv.org/pdf/2104.14294.pdf)

<image src="./images/DINO.png">

DINO的核心思想是通过模仿知识蒸馏的过程，但不使用任何标签，来训练一个模型。适合于那些标签获取成本高昂或难以获得的情况。

<details>
<summary>Dino</summary>

Construct different distorted views, or crops, of an image with multicrop strategy. Contians two global views and several local views of 
smaller resolution. All crops are passed through the student, while only the global views are passed through the teacher. 
对图像构建的不同策略扩增，其中大的作为global view,小的称为local view, global view送入教师网络，local view送入学生网络。
```python
# torchvision.transforms.RandomResizedCrop按照scaimage_sizele的范围切割原图，然后resize到image_size
self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

image_one, image_two = self.augment1(x), self.augment2(x)

local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)
# 这里只送local_crop进学生网络，global_crop送教师网络
student_proj_one, _ = self.student_encoder(local_image_one)
student_proj_two, _ = self.student_encoder(local_image_two)

with torch.no_grad():
    teacher_encoder = self._get_teacher_encoder()
    teacher_proj_one, _ = teacher_encoder(global_image_one)
    teacher_proj_two, _ = teacher_encoder(global_image_two)
```

The exponential moving average (EMA) on the student weight to update teacher weight, update rule is $\theta_t \leftarrow \lambda \theta_t + (1- \lambda ) \theta_s$. $\lambda$ following a cosine schedule. 通过对学生网络参数的指数移动平均（EMA）来更新的教师网络的权重。
```python
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
```

$P_s$和$P_t$ obtained by normalizing the output of network with a softmax function with temperatune controls the sharpness of the output distribution ,$P(x)=\frac{exp(g_{\theta}(x)/\tau)}{\sum^K_{k=1}exp(g_{\theta}(x)/\tau)}$. Match these distributions by minimizing the cross-entropy loss of the student network $min_{\theta_s}-P_t(x)logP_s(t)$. 通过温度参数来锐化网络的输出概率分布。教师网络和学生网络通过预测彼此的输出来进行学习。

```python   
teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

loss_fn_ = partial(loss_fn, student_temp = default(student_temp, self.student_temp),
    teacher_temp = default(teacher_temp, self.teacher_temp), centers = self.teacher_centers)

loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2

def loss_fn(teacher_logits, student_logits, teacher_temp, student_temp, centers, eps = 1e-20 ):
    teacher_logits = teacher_logits.detach()
    #  use temperatune parameter controls the sharpness of the output distribution
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
```
</details>

# Reference
All the code above comes from the aggregative repository https://github.com/lucidrains/vit-pytorch

# 内容延伸

1. [Transforner Base Architecture](./README_transformer.md)

2. [TR Encoder Model ViTs](./README_ViT.md)

3. [Pos-Emb in TR](./README_pos_emb.md)

4. [Visual object track](./README_visual_object_track.md)