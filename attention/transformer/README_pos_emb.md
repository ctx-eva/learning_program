# Position Embedding

参考资料 

(https://blog.csdn.net/v_JULY_v/article/details/134085503)

(https://zhuanlan.zhihu.com/p/650469278)

(https://github.com/ofirpress/attention_with_linear_biases)

(https://developer.aliyun.com/article/842370)

## Sinusoidal position embedding

The sinusoidal position embedding was introduced in transformer model [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

<image src="./images/sin.png">

Sinusoidal position embedding, 利用不同维度上三角函数的周期不同，给不同位序的输入向量分配随位序变化的且唯一的d_model维position向量，以此为模型提供输入向量的位序信息

$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})
$$

```python
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        encodings = self.get_positional_encoding(d_model, max_len)
        # 通过设置persistent=False使得self.encodings独立于模型
        self.register_buffer('positional_encodings', encodings, False)

    @staticmethod
    def get_positional_encoding(d_model: int, max_len: int):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / d_model)) for i in range(d_model)]
                            for p in range(max_len)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return rearrange(pe, '... -> 1 ...')

    def forward(self, x: torch.Tensor):
        # 这里应该不需要detach()和requires_grad_=false,因为已经在register_buffer中设置persistant=false
        pe = self.positional_encodings[:x.shape[1]].detach().requires_grad_(False)
        return self.dropout(x + pe)
```

## Rotary position embedding (RoPE)

Paper: [RoFormer: Enhanced transformer with rotary position embedding](https://arxiv.org/pdf/2104.09864.pdf)

<image src="./images/RoPE.png">

Rotary position embedding 将词向量分成$d_{model}/2$组，对连续两个维度数据进行仿射变换实现仅改变词向量方向而不改变词向量模值，通过对attn计算dot-product时的query和key向量乘以旋转矩阵的方式实现。计算attn矩阵中第$(m,n)$个元素时，
$$Attn_(m,n)=q^T_mk_n=(R^d_{\Theta,m}W_qx_m)^T(R^d_{\Theta,n}W_kx_m)=x^TW^T_q(R^d_{\Theta,m})^TR^d_{\Theta,n}W_kx_n$$
其中$(R^d_{\Theta,m})^TR^d_{\Theta,n}=R^d_{\Theta,n-m}$,$R^d_{\Theta,n-m}$是正交阵且仅和q，k中的词序差相关，向模型提供了词之间的相对位置信息。

序列中第m个词的旋转变换矩阵如下所示:

<image src="./images/RotMat.svg">

由于RoPE仅作用attn的在query和key上，并不作用于value和output上，因此一般对transformer中每层attn都需要添加。

在llama中该变换表示为将词向量的特征分作两两一组，转换到复数域进行向量旋转后，再变换回实数域组合
```python
def precompute_freqs_cis(d_model: int, end: int, theta: float = 10000.0):
    # parameter end here means the seq lens, 但是不应该是在确定模型维度和最大输入长度后就固定freqs_cis的值吗
    # enumerate model demension 计算旋转矩阵中的角度theta的值
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
    # enumerate 序列列举m的值
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # torch.outer实现m中每个元素和theta中每个元素相乘, freqs:[end, d_model//2]
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 将freqs中的角度转化为复数域的表示形式，即cos(m theta)+i*sin(m theta), freqs_cis [end, d_model//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # the x dim here may be as [batch, seq_len, nhead, d_model]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # 如果对输入x做transpose(1, 2)，应该不用做view,因为pytorch中“+，-，*”是broadcastable
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape(*xq.shape[:-1], -1, 2)将输入的最后一维拆成两两一组的形式，
    # torch.view_as_complex返回复数张量形式的输入视图
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 因为freqs_cis的模长是1，旋转角度直接在复数域相乘就可以完成
    # 通过torch.view_as_real将结果转化到实数域，连续两维的值代码一组附加pos embed的结果
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

也可以将向量q,k和旋转矩阵相乘的结果用向量积的形式表示，如下：

<image src="./images/RoFast.svg">

```python
import torch
import math
def rotary_position_embedding(q, k):
    """
    Rotary Position Embedding (RoPE) for queries and keys.
    
    Args:
        q: tensor for queries of shape (batch_size, num_heads, seq_len, dim)
        k: tensor for keys of shape (batch_size, num_heads, seq_len, dim)
        
    Returns:
        Rotated queries and keys
    """
    batch_size, num_heads, seq_len, dim = q.size()
    
    # Begin of sinusoidal_position_embedding content
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)).to(q.device)
    pos_emb = position * div_term
    '''
    the above is same as
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
    t = torch.arange(end, device=freqs.device) 
    freqs = torch.outer(t, freqs).float() 
    Do torch.exp gets some compuate benefit ?
    '''  

    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    pos_emb = pos_emb.unsqueeze(0).unsqueeze(1)
    pos_emb = pos_emb.expand(batch_size, num_heads, -1, -1)
    # End of sinusoidal_position_embedding content

    # Extract and duplicate cosine and sine embeddings
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    """
    the above is same as, due to the broadcastable
    sin_emb = torch.sin(pos_emb).T.flatten()
    cos_emb = torch.cos(pos_emb).T.flatten()
    """

    # Create alternate versions of q and k
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    k_alternate = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.size())

    # Rotate queries and keys
    q_rotated = q * cos_emb + q_alternate * sin_emb
    k_rotated = k * cos_emb + k_alternate * sin_emb

    return q_rotated, k_rotated
```

## Attention with Linear Biases (ALiBi)

Paper: [Train short, test long: Attention with linear biases enables input length extrapolation](https://arxiv.org/pdf/2108.12409)

<image src="./images/Alibi.png">

The following code is organized from (https://github.com/ofirpress/attention_with_linear_biases/issues/5)

```python
def get_slopes(n):
    "calculate slope m of different head"
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
        # 按满足2^a heads的上一个a,先计算2^(-8/a),再从2^(-8/2a)中抽值补齐 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

# 利用python中的broadcastable,对[n,1]和[1,n]矩阵计算结果为[n,n],值为i-j的矩阵
context_position = torch.arange(maxpos)[:, None].cuda()
memory_position = torch.arange(maxpos)[None, :].cuda()
relative_position = memory_position - context_position 
# expand是因为对不同的head需要乘以的slope不同
relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)

# 提供了多种形式
# 1.Symmetric: 即对先于key的query序列按距离进行惩罚，只有forward
self.slopes = torch.Tensor(get_slopes(attn_heads)).cuda()*-1
self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position # 对slope的两次unsqueeze也是为了broadcastable
self.alibi = self.alibi.view(1, attn_heads, maxpos, maxpos)

# 2.Nonsymmetric:
# 构建前向和后向的mask,每个分别占用attn_heads//2
self._future_mask_right = torch.triu(utils.fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1).unsqueeze(0).repeat(attn_heads//2, 1, 1)
self._future_mask_left = torch.tril(utils.fill_with_neg_inf(torch.zeros([maxpos, maxpos])), -1).unsqueeze(0).repeat(attn_heads//2, 1, 1)
self.nonsym_mask = torch.cat((self._future_mask_right, self._future_mask_left), dim = 0).unsqueeze(0).cuda()

# 计算slopes对前向和后向使用相同的slope
self.slopes = torch.Tensor(get_slopes(attn_heads//2)).cuda()*-1

# 和基本的Symmetric处理一样
context_position = torch.arange(maxpos)[:, None].cuda()
memory_position = torch.arange(maxpos)[None, :].cuda()
relative_position = memory_position - context_position
relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads//2, -1,-1)

# 对齐slope和relative_position，以及alibi和dot-product维度
self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
self.alibi = self.alibi.view(1, attn_heads//2, maxpos, maxpos)
self.alibi = self.alibi.repeat(1, 2, 1, 1).cuda()

# 3.Nonsymmetric with no mask
# slopes_left，和slopes_right是可学习的变量
slopes_left = torch.nn.parameter.Parameter(torch.Tensor(attn_heads))
nn.init.normal_(slopes_left, -2,1)
slopes_right = torch.nn.parameter.Parameter(torch.Tensor(attn_heads))
nn.init.normal_(slopes_right, -2,1)

slopes_left = -torch.sigmoid(slopes_left)
slopes_right = -torch.sigmoid(slopes_right)

# 和基本的Symmetric处理一样
context_position = torch.arange(maxpos)[:, None]
memory_position = torch.arange(maxpos)[None, :]
relative_position = memory_position - context_position
relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)

alibi_left = slopes_left.unsqueeze(1).unsqueeze(1) * relative_position
alibi_right = slopes_right.unsqueeze(1).unsqueeze(1) * relative_position

# 对alibi_right保留右上区域，alibi_left保留右下区域，
# 训练时通过这里可以反向传播训练slopes_right,slopes_left, 二者区域互不影响(对角线恒为0)
self.alibi = torch.triu(alibi_right) + torch.tril(alibi_left)

# when appling to dot-product as below, attn_weights is the result of q,k dot-product before softmax
attn_weights += nonsym_mask[:,:,:tgt_len,:src_len].to(attn_weights)
```

## CoPECoPE

# 内容延伸

1. [Transforner Base Architecture](./README_transformer.md)

2. [TR Encoder Model ViTs](./README_ViT.md)

3. [Pos-Emb in TR](./README_pos_emb.md)

4. [Visual object track](./README_visual_object_track.md)

5. [Object Detection based on transformer](./README_detection.md)