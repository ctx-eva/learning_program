# Deformable Convolutional

Paper: [Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf)
$\qquad$ [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/pdf/1811.11168.pdf)
$\qquad$ [An Empirical Study of Spatial Attention Mechanisms in Deep Networks](https://arxiv.org/pdf/1904.05873.pdf)

The original repository link is https://github.com/4uiiurz1/pytorch-deform-conv-v2

Deformable Conv通过对输入层做卷积，先通过旁路卷积计算输出通道数为$ks \times ks$的field offset $\Delta p$,表示每个卷积位置偏置偏离的预测，
再根据 $\Delta p$ 获得参与计算卷积运算位置的值，之后再通过主卷积计算获得输出层
Deformable Conv(变形卷积) 表达形式
一般卷积的表达形式：$y(p_0) = \sum_{p_n\in R}w(p_n)x(p_0 + p_n)$,$p_n$代表了卷积核感受野中相对于卷积位置的偏置。
变形卷积的表达形式：$y(p_0) = \sum_{p_n\in R}w(p_n)x(p_0 + p_n + \Delta p)$,通过$p_n+\Delta p$来改变卷积激活数据的偏置位置。
对$p = p_0 + p_n + \Delta p$所对应的值,$x(p) = x(p_0 + p_n + \Delta p) = \sum_{q}G(q,p)x(q),\sum_{q}G(q,p)x(q)$表示对点p处的值进行双线性插值
Deformable Conv V2 表达形式 $y(p_0) = \sum_{p_n\in R}w(p_n)x(p_0 + p_n + \Delta p)\Delta m$, 在Deformable Conv的基础型上增加了对应位置值的缩放变量 $\Delta m \in (0,1)$

### Deformable Conv 类初始化
self.p_conv 计算偏置offset的旁路卷积，output通道数为$2 \times ks \times ks$，对kernel的每个位置都预测两个偏置值(x,y各一个)
self.m_conv 在Deformable Conv V2中，output通道数为$ks \times ks$，通过self.p_conv计算获得偏置在输入层上双线性插值获得的值每个kernel位置预测一个缩放值。
self.conv 主卷积，计算输出层的feature map, stride=kernel_size表示通过self.p_conv计算获得偏置再双线性插值获得的$(h \times ks,w\times ks)$的feature map,每个$(ks \times ks)$区域计算一次卷积
torch.nn.modules.module.register_module_backward_hook该函数返回一个关于输入的新梯度，该梯度将在随后的计算中代替grad_input，grad_input将只对应于作为位置参数给出的输入。
```python
def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
    """
    Args:
        modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
    """
    super(DeformConv2d, self).__init__()
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.zero_padding = nn.ZeroPad2d(padding)
    
    self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
    nn.init.constant_(self.p_conv.weight, 0)
    self.p_conv.register_backward_hook(self._set_lr)

    self.modulation = modulation
    if modulation:
        self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.m_conv.weight, 0)
        self.m_conv.register_backward_hook(self._set_lr)

@staticmethod
def _set_lr(module, grad_input, grad_output):
    grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
    grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
```

### 计算卷积激活位置
使用self.p_conv计算输入层每个卷积位置上的附加偏置
_get_p_0 根据offset的w,h计算卷积中心位置值，x,y各重复$ks \times ks$次
_get_p_n 根据kernel_size计算初始卷积核偏置，x,y各重复$ks \times ks$次
_get_p计算卷积核每个位置对应的激活位置$p = p_0 + p_n + \Delta p$
```python
def forward(self, x):
    offset = self.p_conv(x)

    dtype = offset.data.type()
    ks = self.kernel_size
    N = offset.size(1) // 2

    if self.padding:
        x = self.zero_padding(x)

def _get_p_n(self, N, dtype):
    p_n_x, p_n_y = torch.meshgrid(
        torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
        torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
    # (2N, 1)
    p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
    p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

    return p_n

def _get_p_0(self, h, w, N, dtype):
    p_0_x, p_0_y = torch.meshgrid(
        torch.arange(1, h*self.stride+1, self.stride),
        torch.arange(1, w*self.stride+1, self.stride))
    p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
    p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
    p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

    return p_0

def _get_p(self, offset, dtype):
    N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

    # (1, 2N, 1, 1)
    p_n = self._get_p_n(N, dtype)
    # (1, 2N, h, w)
    p_0 = self._get_p_0(h, w, N, dtype)
    p = p_0 + p_n + offset
    return p

    # (b, 2N, h, w)
    p = self._get_p(offset, dtype)
```

### 双线性插值计算激活位置在输入层上的值
p.contiguous().permute(0, 2, 3, 1)将p转成连续内存将第一维转到最后一维，维度为 $(b,h,w,2 \times ks \times ks)$
激活位置$p$,取其x,y对应的floor,ceil四角坐标在输入层上的值，使用torch.clamp排除在输入层区域外的值，$p$前N个值对应x坐标，后N个值对应y坐标，$N = ks \times ks $
双线新插值的公式如下:
<image src="./images/linear_ interpolation_formula.svg" style="zoom:200%">
式中: $Q_{lt}$ , $Q_{lb}$ , $Q_{rt}$ , $Q_{rb}$ 分别表示插值点左上，左下，右上，右下的点，$ x_r-x_l , y_b - y_t $ 表示x,y方向插值采样点间隔值,在这里 $x_r-x_l=1,y_b-y_t=1$, 
同时可得 $x_r-x_p = 1 + (x_l-x_p), y_b-y_p = 1 + (y_t - y_p), x_p-x_l=1-(x_r-x_p), y_p-y_t=1-(y_b-y_p)$,分别计算出左上，左下，右上，右下方向上的插值权重 $G(q,p)$ 。
_get_x_q 根据q点坐标在输入层上取值，得到x_offset, $ dim = (b,input_channels,h,w,ks \times ks )$ , 根据插值权重 $G(q,p)$ 对各个方向上的x_offset进行加权求和。

``` python
# (b, h, w, 2N)
p = p.contiguous().permute(0, 2, 3, 1)
q_lt = p.detach().floor()
q_rb = q_lt + 1

q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

# clip p
p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

# bilinear kernel (b, h, w, N)
g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

def _get_x_q(self, x, q, N):
    b, h, w, _ = q.size()
    padded_w = x.size(3)
    # c = input channels
    c = x.size(1)
    # (b, c, h*w)
    x = x.contiguous().view(b, c, -1)

    # (b, h, w, N)
    index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
    # (b, c, h*w*N)
    index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

    x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

    return x_offset

# (b, c, h, w, N)
x_q_lt = self._get_x_q(x, q_lt, N)
x_q_rb = self._get_x_q(x, q_rb, N)
x_q_lb = self._get_x_q(x, q_lb, N)
x_q_rt = self._get_x_q(x, q_rt, N)

# (b, c, h, w, N)
x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
            g_rb.unsqueeze(dim=1) * x_q_rb + \
            g_lb.unsqueeze(dim=1) * x_q_lb + \
            g_rt.unsqueeze(dim=1) * x_q_rt
```

### deform conv V2 对卷积和各方向的值进行缩放权重
m = torch.sigmoid(self.m_conv(x))， self.m_conv(x)对输入层做卷积输出 $dim=(b,ks \times ks, h, w)$ , 用torch.sigmoid将系数限制在0,1之间。
对m在第二维上重复input_channels次，和x_offset统一维度相乘。
```python
if self.modulation:
    m = torch.sigmoid(self.m_conv(x))
# modulation
if self.modulation:
    m = m.contiguous().permute(0, 2, 3, 1)
    m = m.unsqueeze(dim=1)
    m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
    x_offset *= m
```

### 变换offset维度，通过主卷积，完成该层Deform conv
先将offset维度变换成 $dim = (b, input_channels, h, w \times ks, ks)$, 再变换成 $dim = (b, input_channels, h \times ks, w \times ks)$
offset转变成卷积核大小为kernel_size,stride为kernel_size的主卷积self.conv对应的输入层。
```python
@staticmethod
def _reshape_x_offset(x_offset, ks):
    b, c, h, w, N = x_offset.size()
    x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
    x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

    return x_offset

x_offset = self._reshape_x_offset(x_offset, ks)
out = self.conv(x_offset)

return out
```

