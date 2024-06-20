# Spatial Transformer Networks with non-linear thin-plate-spline (TPS) transformation

Paper: [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)

$\qquad$ [Principal Warps: Thin-Plate Splines and Decomposition of Deformations](https://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf)

$\qquad$ [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)

$\qquad$ [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](https://sci-hub.se/10.1109/TPAMI.2018.2848939)

the code is forked from original repository link https://https://github.com/ayumiymk/aster.pytorch/tree/master/lib/models

原始版本的Spatial Transformer Networks 结构如下图所示:

<image src="./images/STN.webp">

首先通过一个浅层网络Localization Network 估计 crop Matrix, Affine Matrix 或者 Perspective Matrix 的变换矩阵，该矩阵表示Output: Region of interest 和 Input: Full Image 之间的图像变换矩阵。接下来，通过 Grid generate 计算在 Region of interest 上每个Grid坐标变换到 Full Image 中的像素坐标。根据变换后的坐标利用双线性插值在 Full Image 中进行Sample获得像素值生成Region of interest。

Spatial Transformer Networks in RARE and ASTER 中，计算 Grid generate 时采用薄板样条采样，而不是之前的变换矩阵。

## thin-plate-spline (薄板样条采样) 

### 回顾基础知识

1. 双调和方程(Biharmonic equation)

   写做：
   
   <image src="./images/Biharmonic_eq.svg">

   其中：

   <image src="./images/nabla_opt.svg">哈密尔顿算子：<image src="./images/nabla.svg">，表示物理量在坐标方向的偏导数的和。

   <image src="./images/grad_opt.svg">梯度：<image src="./images/grad.svg">，表示<image src="./images/nabla_opt.svg">作用于标量方程 $ \varphi $, 标量方程的梯度是一个矢量场

   <image src="./images/div_opt.svg">散度：<image src="./images/div.svg">表示区域流入流出的矢量多少，<image src="./images/nabla_opt.svg">和矢量的点乘结果是标量。

   <image src="./images/Laplace_opt.svg">拉普拉斯算子：<image src="./images/Laplace.svg">，表示梯度的散度。

   在2D直角坐标系下，双调和方程有如下形式：

   <image src="./images/2D_Biharmonic_eq.svg">

   推导过程如下：

   <image src="./images/2D_Biharmonic_eq_infer.svg">

2. 薄板样条插值函数推导：

   薄板样条目标是使得薄板能量方程值最小：<image src="./images/EnergyEqn.svg">

   使用参数 $\lambda$ 控制薄板变形强度，获得最优的拟合平滑方程：

   <image src="./images/EnergySmoothEq.svg">

   使上式最小 <image src="./images/mineq.svg">，<image src="./images/minEnergy.svg">

   构建方程<image src="./images/Uxy.svg">满足上述条件

   $U(x,y)$ 满足条件推导

   一阶偏导<image src="./images/1th_partial.svg"> 

   二阶偏导<image src="./images/2th_partial.svg">

   三阶偏导<image src="./images/3th_partial.svg">

   四阶偏导<image src="./images/4th_partial.svg">

3. 薄板样条插值函数的插值计算

   由平滑方程，构建薄板样条变换方程如下：

   <image src="./images/TPS_tf.svg">

   在Spatial Transformer Networks中要计算Output中每个grid的x,y坐标在Input原图像中的坐标位置。经由STN估计出的Input图像空间中的基准点坐标 $(x,y)$ 后，和在Output图像空间中的基准点 $(x',y')$ 一一对应，可以列出如下两组方程：

   <image src="./images/TPS_SimuEq.svg">
   
   增加三个约束方程

   <image src="./images/TPS_constraint_eq.svg">

   构建如下方程组矩阵

   <image src="./images/matrix.svg">

   令矩阵  <image src="./images/L_matrix.svg"> 可得 $V = LW,W=L^{-1}V$ 。

   矩阵向量 $W$ 表示求得的薄板样条变换方程中的各项系数，可以计算在Output图像空间的任意坐标点 $(x',y')$ ，变换到Input图像空间的位置坐标 $(x,y)$

### TPSSpatialTransformer 薄板样条采样变换初始化

```python
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

class TPSSpatialTransformer(nn.Module):

def __init__(self, output_image_size=None, num_control_points=None, margins=None):
    super(TPSSpatialTransformer, self).__init__()
    self.output_image_size = output_image_size
    self.num_control_points = num_control_points
    self.margins = margins

    self.target_height, self.target_width = output_image_size
```

### 生成在Output上的基准点坐标

num_control_points表示一共用到的基准点个数，由于期望将文字区域的上下边缘对齐到Output，上下边缘上各安排1/2的基准点，margin表示着各个边缘方向的缩进。用np.linspace生成均匀间隔的基准点序列，生成的坐标(x,y)值被归一化到(0,1)之间。

```python
    target_control_points = build_output_control_points(num_control_points, margins)

    # output_ctrl_pts are specified, according to our task.
    def build_output_control_points(num_control_points, margins):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # ctrl_pts_top = ctrl_pts_top[1:-1,:]
        # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
        output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
        return output_ctrl_pts
```

### 构建变换矩阵L

变换矩阵 $L$ 的维度为 $(N+3) \times (N+3)$, 通过torch.zeros(N + 3, N + 3)初始化，同时完成右下角全零矩阵 $O$ 的构建。compute_partial_repr用来计算两组点列间的 $U(r_{ij})$, 通过 input_points.view(N, 1, 2) - control_points.view(1, M, 2) 实现生成一个维度为 $N \times M$ 的坐标偏差值。通过平方，求和等构建 $U = r^2logr^2$ 的结果。当输入的两组点列为Output空间上的基准点，compute_partial_repr 可以获得变换矩阵L分块中的左上角 $N \times N$ 矩阵 $K$ 。通过fill_，copy_对右上坐下的矩阵 $P,P^T$ 的赋值，最终生成变换矩阵 $L$ ，并对 $L$ 求逆以备在forward中求解变换方程系数矩阵 $W$ 。

```python    
    N = num_control_points
    # N = N - 4

    # create padded kernel matrix
    forward_kernel = torch.zeros(N + 3, N + 3)
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)

    def compute_partial_repr(input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix
    

    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

    # compute inverse matrix
    inverse_kernel = torch.inverse(forward_kernel)
```

### 根据Output的尺寸构建grid,并计算在TPS变换方程下的各部分矩阵

list(itertools.product(range(self.target_height), range(self.target_width))) 计算Output空间中grid的x,y坐标，效果和np.meshgrid()+np.stack()类似。各除以宽高，归一化到区间(0,1)。这里的compute_partial_repr用来计算Output空间中每个grid坐标到控制点的 $U(r_{grid_i,j})$ ，构建维度为 $NumGrid \times NumCtrlP$ 的 $K'$ 矩阵。torch.ones(HW, 1), target_coordinate 用来表示新的 $P'$ 矩阵。

```python
    # create target cordinate matrix
    HW = self.target_height * self.target_width
    target_coordinate = list(itertools.product(range(self.target_height), range(self.target_width)))
    target_coordinate = torch.Tensor(target_coordinate) # HW x 2
    Y, X = target_coordinate.split(1, dim = 1)
    Y = Y / (self.target_height - 1)
    X = X / (self.target_width - 1)
    target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([
    target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
    ], dim = 1)
```

注册要用到的变量

```python
    # register precomputed matrices
    self.register_buffer('inverse_kernel', inverse_kernel)
    self.register_buffer('padding_matrix', torch.zeros(3, 2))
    self.register_buffer('target_coordinate_repr', target_coordinate_repr)
    self.register_buffer('target_control_points', target_control_points)
```

### 计算Output空间中Grid坐标在Input空间中的差值点坐标

利用L的逆矩阵和Localization Network预测得到的Input空间基准点坐标计算TPS变换方程的系数矩阵 $W$ ,即mapping_matrix。使用矩阵乘法点乘self.target_coordinate_repr和mapping_matrix获得Output空间中Grid坐标在Input空间中的差值点坐标。

```python
def forward(self, input, source_control_points):
    assert source_control_points.ndimension() == 3
    assert source_control_points.size(1) == self.num_control_points
    assert source_control_points.size(2) == 2
    batch_size = source_control_points.size(0)

    Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
    mapping_matrix = torch.matmul(self.inverse_kernel, Y)
    source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
```

### 插值计算Output 

使用torch.nn.functional.grid_sample,根据插值点坐标source_coordinate，获取对应的像素值。因为Input上的像素点坐标在(0,1)区间内，但这里没有对变换后的Grid坐标进行严格限制，需要用torch.clamp限制grid的输出坐标。由于grid_sample函数直接对接原始的STN网络，其输出使用tanh激活函数，输出结果在(-1,1)区间上，因此grid_sample函数接收的插值输入是(-1,1)区间的坐标，(0，0)表示图像中心，图像左上坐标表示为（-1，-1），需要将Grid坐标变换到（-1,1）区间内。

```python
    grid = source_coordinate.view(-1, self.target_height, self.target_width, 2)
    grid = torch.clamp(grid, 0, 1) # the source_control_points may be out of [0, 1].
    # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
    grid = 2.0 * grid - 1.0
    output_maps = grid_sample(input, grid, canvas=None)

    def grid_sample(input, grid, canvas = None):
        output = F.grid_sample(input, grid)
        if canvas is None:
            return output
        else:
            input_mask = input.data.new(input.size()).fill_(1)
            output_mask = F.grid_sample(input_mask, grid)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output
    
    return output_maps, source_coordinate
```

## Spatial Transformer Network

### Spatial Transformer Network 初始化及其权重初始化

```python
import math
import numpy as np
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .tps_spatial_transformer import TPSSpatialTransformer

def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
        conv_layer,
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        )
    return block


class STNHead(nn.Module):
    def __init__(self, in_planes, num_ctrlpoints, tps_inputsize, activation='none', output_image_size=None, num_control_points=None, margins=None):
        super(STNHead, self).__init__()

        self.in_planes = in_planes
        self.num_ctrlpoints = num_ctrlpoints
        self.tps_inputsize = tps_inputsize
        self.activation = activation
        self.stn_convnet = nn.Sequential(
                              conv3x3_block(in_planes, 32), # 32*64
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              conv3x3_block(32, 64), # 16*32
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              conv3x3_block(64, 128), # 8*16
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              conv3x3_block(128, 256), # 4*8
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              conv3x3_block(256, 256), # 2*4,
                              nn.MaxPool2d(kernel_size=2, stride=2),
                              conv3x3_block(256, 256)) # 1*2

        self.stn_fc1 = nn.Sequential(
                          nn.Linear(2*256, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(512, num_ctrlpoints*2)

        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(output_image_size),
            num_control_points=num_control_points,
            margins=tuple(margins))

        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)
```

初始化Localization Network中的cnn layers

```python
    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
```

初始化Localization Network中全连接层，由于需要网络backpropagation来调整Localization网络的权重，若随机初始化网络会导致一开始网络不能接收到正常的图像，影响网络的收敛性。将weight设为全0，bias设为Output上的基准点相同值，可以保证网络在一开始训练过程中能够接收到大部分图像。

```python
    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1.-margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1-margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation is 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)
```

在ASTER中为了缩减计算，对Localization网络的输入做了下采样，采用torch.nn.functional.interpolate获得下采样插值到tps_inputsize的图像，通过Localization网络计算得到Input上的基准点ctrl_points。获得基准点后，通过TPSSpatialTransformer获得Output的插值点坐标，为了保持插值后Output图像的清晰度，采用原Input图像计算插值点像素。

```python
    def forward(self, x):
        x_input = x
        x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        ctrl_points = x.view(-1, self.num_ctrlpoints, 2)
        x, _ = self.tps(x_input, ctrl_points)
        return x
```
