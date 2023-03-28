
import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedBCELoss(nn.Module):
    """
    This code revised from the focal loss in yolov8
            https://github.com/ultralytics/ultralytics/yolo/utils/metrics.py
    """
    def __init__(self, alpha=0.75, reduction='mean'):
        super(BalancedBCELoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        loss *= alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLoss(nn.Module):
    """
    This code referenced to
            https://github.com/ultralytics/ultralytics/yolo/utils/metrics.py
    """
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

class QFocalLoss(nn.Module):
    """
    This code referenced to
            https://github.com/ultralytics/yolov5/utils/loss.py
    """
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

class VarifocalLoss(nn.Module):
    """
    This code revised from the dfl_loss in yolov8
            https://github.com/ultralytics/ultralytics/yolo/utils/loss.py
    """
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self, gamma=2.0, alpha=0.75, reduction='sum'):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, gt_score):#, label):
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

class DFocalLoss(nn.Module):
    """
    This code revised from the dfl_loss in yolov8
            https://github.com/ultralytics/ultralytics/yolo/utils/loss.py
    """
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
