#损失函数总结

##Binary Cross Entropy loss (BCEloss)

![BCE_loss](https://latex.codecogs.com/svg.image?\mathbf{BCE}_{loss}=-[ \nu_{gt} * \log (\nu_{pred}) + (1 - \nu_{gt}) * \log (1 - \nu_{pred}) ])

当 $\nu_{gt}$ 是 one-hot 类型时

$
\mathbf{BCE}_{loss}=
\begin{cases}
 \log (\nu_{pred}), &if\ \nu_{gt}\ = 1\\
\log (1 - \nu_{pred}), &if\  \nu_{gt}\ = 0
\end{cases}
$

```python
import torch
import torch.nn as nn
#sigmoid将output的值映射到(0,1)区间
m = nn.Sigmoid()
criterion = nn.BCELoss()
criterion(m(output), target)

#BCEWithLogitsLoss在计算loss之间已经对output做了sigmoid操作
criterion = torch.nn.BCEWithLogitsLoss()
criterion(output, target)
```

###Balanced BCE

![Balanced BCE_loss](https://latex.codecogs.com/svg.image?\mathbf{BCE}_{loss}^{Balanced}=-[ \alpha * \nu_{gt} * \log (\nu_{pred}) + (1 - \alpha) (1 - \nu_{gt}) * \log (1 - \nu_{pred}) ]

当 $\nu_{gt}$ 是 one-hot 类型时

$
\begin{aligned}
\mathbf{BCE}_{loss}^{Balanced} &= - [ (\alpha * \nu_{gt} + (1 - \alpha) (1 - \nu_{gt}) )* \log (\nu_{pred}) + (\alpha * \nu_{gt} + (1 - \alpha) (1 - \nu_{gt}) )* \log (1 - \nu_{pred}) ] \\
&= [\alpha * \nu_{gt} + (1 - \alpha) (1 - \nu_{gt}) ] * \mathbf{BCE}_{loss}
\end{aligned}
$

Balanced BCE 通过 $\alpha$ 控制正负样本的加权参数,改变正负样本参与loss计算的贡献比例,对目标检测类任务通过 $\alpha$ 平衡正负样本间的数量差异.

```python
import torch
import torch.nn as nn
"""
This code revised from the focal loss in yolov8
           https://github.com/ultralytics/ultralytics/yolo/utils/metrics.py
"""
class BalancedBCELoss(nn.Module):
    def __init__(self, alpha=0.75, reduction='mean'):
        super(BalancedBCELoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        loss *= alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
```

##Focal loss (FL) 

paper: (https://arxiv.org/pdf/1708.02002.pdf)

$
\begin{aligned}
\mathbf{FL} &= - [ \nu_{gt} * (1 - \nu_{pred})^{\gamma} * \log (\nu_{pred}) + (1 - \nu_{gt}) * \nu_{pred}^{\gamma} * \log (1 - \nu_{pred}) ] \\
&= - [ \nu_{gt} * (1 - \nu_{pred})^{\gamma} * \log (\nu_{pred}) + (1 - \nu_{gt}) * (1 - ( 1 - \nu_{pred}))^{\gamma} * \log (1 - \nu_{pred}) ]
\end{aligned}
$

当 $\nu_{gt}$ 是 one-hot 类型时

$
\begin{aligned}
\mathbf{FL} &= - (1 - \nu_{t})^{\gamma} * \log (\nu_{t}) = (1 - \nu_{t})^{\gamma} * \mathbf{BCE}_{loss} \ , \ \nu_{t} =
\begin{cases}
\nu_{pred}, &if\ \nu_{gt}\ = 1\\[2ex]
1 - \nu_{pred}, &if\  \nu_{gt}\ = 0
\end{cases} \\
&= [1 - (\nu_{gt} * \nu_{pred} + ( 1 - \nu_{gt} ) * ( 1 - \nu_{pred})) ]^{\gamma} * \mathbf{BCE}_{loss}
\end{aligned}
$

加入 $\alpha$ 正负样本均衡后,可表示如下:

$
\mathbf{FL} = [\alpha * \nu_{gt} + (1 - \alpha) (1 - \nu_{gt}) ] * [1 - (\nu_{gt} * \nu_{pred} + ( 1 - \nu_{gt} ) * ( 1 - \nu_{pred})) ]^{\gamma} * \mathbf{BCE}_{loss}
$

<!-- ![Focal loss 形式](https://img-blog.csdnimg.cn/dd83fc4d77944c589941fc08b5d6c889.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmlnSGFvNjg4,size_20,color_FFFFFF,t_70,g_se,x_16) -->
![Focal loss 形式](img/FocalLoss.png)

Focal loss 通过 $(1 - \nu_{t})^{\gamma}$ 控制难易样本参与loss计算的贡献比例, 对于正样本易分样本 $\nu_{pred}$ 越接近1, $(1 - \nu_{pred})^{\gamma}$ 接近0, 参与loss比例越小; 难分样本, $\nu_{pred}$ 越接近0, $(1 - \nu_{pred})^{\gamma}$ 接近1, 参与loss比例越大. 同样对于负样本, 易分样本 $\nu_{pred}$ 越接近0, $(1 - ( 1 - \nu_{pred}))^{\gamma} = \nu_{pred}^{\gamma}$ 接近0, 参与loss比例越小; 相反负样本的难分样本 $\nu_{pred}^{\gamma}$ 接近1, 参与loss比例越大.

当参数 $\gamma$ 越接近0时, Focal loss 越接近BCEloss.

```python
import torch
import torch.nn as nn
"""
This code referenced to
           https://github.com/ultralytics/ultralytics/yolo/utils/metrics.py
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
```

##Generalized Focal Loss (GFL) 

paper:(https://arxiv.org/pdf/2006.04388.pdf)

###Quality Focal Loss (QFL)

$
\mathbf{QFL} = -\left| \nu_{gt} - \sigma(\nu_{pred}) \right|^{\beta} * [(1 - \nu_{gt}) * \log(1 - \sigma(\nu_{pred})) +  \nu_{gt} * \log(\sigma(\nu_{pred}))]
$

在QFL中负样本的真值 $\nu_{gt} = 0$, 正样本真值 $\nu_{gt} \in [0,1]$ 是0~1之间的概率值.


<image src="img/QualityFocalLoss.png">

图:当 $\nu_{gt} = 0.5$ 时, $-\left| \nu_{gt} - \sigma(\nu_{pred}) \right|^{\beta}$ 的变化趋势
<br/>


当真值 $\nu_{gt} \in {0,1}$ 是 one-hot类型时, QFL和FL具有相同的形式.

$
\mathbf{QFL} = \left| \nu_{gt} - \sigma(\nu_{pred}) \right|^{\beta} * \mathbf{BCE}_{loss} = \begin{cases}
[1 - \sigma(\nu_{pred})]^{\beta} * \mathbf{BCE}_{loss}, &if\ \nu_{gt}\ = 1\\[2ex]
\sigma(\nu_{pred})^{\beta} * \mathbf{BCE}_{loss}, &if\  \nu_{gt}\ = 0
\end{cases} = (1 - \nu_{t})^{\beta} * \mathbf{BCE}_{loss} , \nu_{t} = \begin{cases}
\sigma(\nu_{pred}) , &if\ \nu_{gt}\ = 1\\[2ex]
1 - \sigma(\nu_{pred}), &if\  \nu_{gt}\ = 0
\end{cases} = \mathbf{FL}
$

```python
import torch
import torch.nn as nn
"""
This code referenced to
           https://github.com/ultralytics/yolov5/utils/loss.py
"""
class QFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75, reduction='mean'):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
```
###Distribution Focal Loss (DFL)

Distribution Focal Loss 在 Generalized Focal Loss 中被用作 box_regression. 求取offset形式的边界框(t,l,b,t),这里将 box_regression 问题中边界框的估计视作等效的脉冲响应的概率分布.有 $\int_\infty^\infty \delta(x-y)xdx = 1$ . 将对值的估计范围限制在 $[x_0,x_n]$ 之间,并且令分隔间隔等于1,最终对值的估计可以视为 

$\hat{y} = \int_\infty^\infty \delta(x-y)xdx \sim \int_{y_0}^{y_n} P(x_i)x_i = \sum_{i=0}^n P(x_i)x_i $, $ P(x_i) $ 表示在 $x_i$ 处对 $\hat{y}$ 的概率估计,且有 $\sum^n_{i=0} P(x_i) = 1$. 通过设定分度将边界的估计问题转化为对边界值的分布概率的估计问题.

$\mathbf{DFL}(\mathcal{S}_i,\mathcal{S}_{i+1}) = -((y_{i+1} - y)\log(\mathcal{S}_{i+1}) + (y - y_i)\log(\mathcal{S}_{i})) \ , \ \mathcal{S}_i=\frac{y_{i+1}-y}{y_{i+1}-y_i}$，$\mathcal{S}_{i+1}=\frac{y - y_i}{y_{i+1}-y_i}$

DFL的优化目标使得 $\hat{y}$ 概率映射到 $ceil(y)$ 和 $floor(y)$ 的线性加权和最小

```python
import torch
import torch.nn.functional as F
"""
This code revised from the dfl_loss in yolov8
           https://github.com/ultralytics/ultralytics/yolo/utils/loss.py
"""
class DFocalLoss(nn.Module):
    def __init__(self, ):
        super(DFocalLoss, self).__init__()

    def forward(pred_dist, target):  
        # pred_dist : num_select_anchors*4*reg分度， target : num_select_anchors×4
        # num_select_anchors = num_target_all_batch * select_topk
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
```


**torch.nn.functional.cross_entropy 
注意1:input不需要经过softmax,直接从fn层拿出来的张量就可以送入交叉熵中,因为在交叉熵中已经对输入input做了softmax了. 
注意2:不用对target进行one_hot编码,因为nll_loss函数已经实现了类似one-hot过程. 
referenced to https://blog.csdn.net/qq_38308388/article/details/121640312**

##Varifocal Loss (VFL) 

paper:(https://arxiv.org/pdf/2008.13367.pdf)

$
\mathbf{VFL} = \begin{cases}
-\nu_{gt-score}*[ \nu_{gt-score} * \log (\nu_{pred}) + (1 - \nu_{gt-score}) * \log (1 - \nu_{pred}) ], &if\ \nu_{gt-score}\ \gt 0\\[2ex]
-\alpha*\nu_{pred}^\gamma*\log (1 - \nu_{pred}), &if\  \nu_{gt-score}\ = 0
\end{cases} = [\alpha*\nu_{pred}^\gamma*(1 - (\nu_{gt-score} >0)) + \nu_{gt-score}]* \mathbf{BCE}_{loss}(\nu_{pred},\nu_{gt-score})
$

VFL以IoU-Aware Classification Score(IACS)作为优化目标, $\nu_{gt-score}$ 是pred_box和gt_box的IOU * $\nu_{gt}$. 

**注意:IOU和gt_cls生产过程如下:
1.IOU表示每个batch的pred_boxes和gt_boxes的交并比,对于batch>1的训练过程来说,IOU的维度为 $dim(batch, num\_anchors, max\_num\_gt)$, max_num_gt表示batch中image拥有最大的gt_boxes.
2.需要确定pred_boxes对gt_boxes的归属问题,采用center_belongs_to_grid或者tal的策略确定每个pred_boxes属于哪个gt_boxes获得target_gt_idx,将IOU的维度转化为 $iou_{trans}$ 维度为 $dim(batch, num\_anchors,1)$, $\nu_{gt}$ 的维度为 $dim(batch, num\_anchors, num\_classes)$, 可得到 $\nu_{gt-score} = \nu_{gt} * iou_{trans}$**

```python
import torch
import torch.nn.functional as F
"""
This code revised from the dfl_loss in yolov8
           https://github.com/ultralytics/ultralytics/yolo/utils/loss.py
"""
class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self, gamma=2.0, alpha=0.75, reduction='sum'):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, gt_score):, label):
        pred = pred.sigmoid()
        weight = self.alpha * pred.pow(self.gamma) * (1 - gt_score.ge(0).float()) + gt_score
        # weight = self.alpha * pred.pow(self.gamma) * (1 - label) + gt_score * label
        # with torch.cuda.amp.autocast(enabled=False):
        loss = self.loss_fcn(pred, gt_score) * weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
```

VFL要解决的问题是,在目标检测中正负样本不均衡,负样本的数量远远大于正样本,通过 $\nu_{pred}^\gamma$ 来削减负样本对结果的影响,对负样本其估计越好,$\nu_{pred}^\gamma$ 值越小,即对越容易分类的负样本给予越低的权重,对越难估计的负样本给予越高的权重,更关注对于难估计的负样本的调整.对于正样本使用参数 $\nu_{gt}$ ,对框iou更大的目标给予更大的权重,使得网络更加关注对iou高的预测框的调整.

##多分类损失函数

torch中单分类和多分类的损失没有什么重大的分别

**注意:在多分类的时候，我们希望输出是符合概率分布,问题即转化成为对于网络输出如何处理上.常见的有对输出做sigmoid或者softmax.二者均能把输出转换到(0,1)的区间内.但二者目的不同,对于softmax操作是考虑目标分类严格的参照 $\sum^n_{i=0} P(x_i) = 1 $, 例如在DFL对边界值的分布情况还有单个数字识别中的分类就存在这样的情况;对于sigmoid操作,更加关注当前 $\nu_{pred}$ 是否接近 $\nu_{gt}$,比如对一般的目标识别网络,不能简单的将所有分类互斥作为条件带入.**

##IOU Loss

###IOU Loss

$\mathbf{IOU} = \frac{Intersection(b^{pred},b^{gt})}{Union(b^{pred},b^{gt})} = \frac{Intersection(b^{pred},b^{gt})}{\mathcal{S}^{pred} + \mathcal{S}^{gt} - Intersection(b^{pred},b^{gt})}$

$ Intersection(b^{pred},b^{gt}) = \mathbf{maximum}\left(\mathbf{minimum}(b^{pred}_r,b^{gt}_r)-\mathbf{maximum}(b^{pred}_l,b^{gt}_l),0 \right) * \mathbf{maximum}\left(\mathbf{minimum}(b^{pred}_b,b^{gt}_b)-\mathbf{maximum}(b^{pred}_t,b^{gt}_t),0\right) = \mathbf{I}_w * \mathbf{I}_h$

$\cal{L}_{IOU}$ (IOU_loss)是anchor_pred的IOU和 $\nu_{gt}$ 的交叉熵, $\cal{L}_{IOU} = \nu_{gt} * \log (IOU) + (1 - \nu_{gt}) * \log (1 - IOU)$

####IOU backpropagation

paper:(https://arxiv.org/pdf/1608.01471.pdf)

IOU_loss的反向传播需要计算 $b^{pred}$ 对于 $\cal{L}_{IOU}$ 中各项的偏导.

$\frac{\partial{\mathcal{S}^{pred}}}{\partial{b^{pred}_r}\ (\mathbf{or} \ \partial{b^{pred}_l})} = b^{pred}_b - b^{pred}_t \ ,\ \frac{\partial{\mathcal{S}^{pred}}}{\partial{b^{pred}_t}\ (\mathbf{or} \ \partial{b^{pred}_b})} = b^{pred}_r - b^{pred}_l $

$\frac{\partial{Intersection}}{\partial{b^{pred}_r}\ (\mathbf{or} \ \partial{b^{pred}_l})} = \begin{cases} \mathbf{I}_h ,&if\ b^{pred}_r < b^{gt}_r \ (\mathbf{or} \ b^{pred}_l > b^{gt}_l )\\[2ex] 0 ,&otherwise \end{cases}$ , $\frac{\partial{Intersection}}{\partial{b^{pred}_t}\ (\mathbf{or} \ \partial{b^{pred}_b})} = \begin{cases} \mathbf{I}_w ,&if\ b^{pred}_t > b^{gt}_t \ (\mathbf{or} \ b^{pred}_b < b^{gt}_b )\\[2ex] 0 ,&otherwise \end{cases}$

###GIOU

paper:(https://arxiv.org/pdf/1902.09630.pdf)

$\mathbf{GIOU} = \mathbf{IOU} - \frac{A^c-Union}{A^c}$

$A^c = \left(\mathbf{maximum}(b^{pred}_r,b^{gt}_r)-\mathbf{minimum}(b^{pred}_l,b^{gt}_l)\right) * \left(\mathbf{maximum}(b^{pred}_b,b^{gt}_b)-\mathbf{minimum}(b^{pred}_t,b^{gt}_t)\right) $

$\cal{L}_{GIOU} = 1 - \mathbf{GIOU} \ \in[0,2]$

**GIOU_Loss加入非重合区域的影响，当IOU值相同时，非重合区域占比越小，代表预测框与目标框的对比效果越好。**

###DIOU

paper:(https://arxiv.org/pdf/1911.08287v1.pdf)

$\mathbf{DIOU} = \mathbf{IOU} - \left(\frac{\rho^2({bc}^{pred},{bc}^{gt})}{c^2}\right) = \mathbf{IOU} - \left(\frac{d^2}{c^2}\right)$

${bc}^{pred}$,${bc}^{gt}$ 表示pred_box和gt_box的中心点.d代表pred_box和gt_box的中心点距离,c代表pred_box和gt_box的最小外接矩形对角线长度。

$\cal{L}_{DIOU} = 1 - \mathbf{DIOU} \ \in[0,2]$

**DIOU_Loss用中心点的归一化距离代替了GIOU中的非重合区域占比指标,可以直接最小化两个目标框的距离，比GIOU收敛的更快.在目标框和预测框相互包裹的条件下，DIOU_Loss可以使回归非常快，而GIOU_Loss几乎退化为IOU Loss.**

###CIOU

paper:(https://arxiv.org/pdf/2005.03572.pdf)

$\mathbf{CIOU} = \mathbf{IOU} - \left(\frac{\rho^2({bc}^{pred},{bc}^{gt})}{c^2}+\alpha\nu\right),\ \nu = \frac{4}{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h})^2,\ \alpha=\frac{\nu}{1-IOU+\nu}$

$\cal{L}_{CIOU} = 1 - \mathbf{CIOU} \ \in[0,2]$

**CIoU在DIoU的基础上增加了检测框尺度的loss，增加了长和宽的loss，使得预测框就会更加的符合真实框.CIOU使得评估更加准确,但增加了loss的计算量.**

```python
import torch
"""
This code referenced to
           https://github.com/ultralytics/ultralytics/yolo/utils/metrics.py
"""
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
```

##Smooth L1 loss

$J = \frac{1}{N}\sum^{N}_{i=1}\mathcal{Smooth\ L1}\left(\nu_{pred}-\nu_{gt}\right) = \begin{cases}0.5(\nu_{pred}-\nu_{gt})^2/\beta ,\ &\left|\nu_{pred}-\nu_{gt}\right| < \beta \\ \left|\nu_{pred}-\nu_{gt}\right| - 0.5\beta,\ &Otherwise\end{cases}$

<image src="img/smooth_L1.png">