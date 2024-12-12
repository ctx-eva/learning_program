# Pixel-IOU calculate & assigner

文件结构一览
```
.
|(calculating piou)
├-- box_intersection_2d.py
├-- box_iou_rotated_diff.py
├-- rotated_iou_loss.py
|
|(target align assign for piou)
|
`-- Readme.md
```

## Pixel-IOU calculate

The original repository link is https://github.com/zf020114/DARDet

### 由 $(x,y,w,h,\theta)$ 表示的rectangle，计算四角顶点
<image src="../images/oriented_bounding_boxes.png">

```python
def rotated_box_to_poly(rotated_boxes: torch.Tensor):
    """ Transform rotated boxes to the Quadrangular vertex polygons
    Args:
        rotated_boxes (Tensor): (x, y, w, h, a) with shape (n, 5)
    Return:
        polys (Tensor): 4 corner points (x, y) of polygons with shape (n, 4, 2)
    """
    cs = torch.cos(rotated_boxes[:, 4])
    ss = torch.sin(rotated_boxes[:, 4])
    w = rotated_boxes[:, 2] - 1
    h = rotated_boxes[:, 3] - 1

    x_ctr = rotated_boxes[:, 0]
    y_ctr = rotated_boxes[:, 1]

    #参考上图theta表示box俺顺时针方向旋转角度
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)    #x1 = xc + w/2*cos(theta) + h/2*sin(theta)
    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)    #y1 = xc + w/2*sin(theta) - h/2*cos(theta)
    #（x1,y1）表示旋转之前（xmax,ymin）右上的点对应旋转之后的值

    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0) 
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)   #（x2,y2）表示旋转之前（xmax,ymax）右下的点对应旋转之后的值

    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)  #（x3,y3）表示旋转之前（xmin,ymax）左下的点对应旋转之后的值
    
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0) #（x4,y4）表示旋转之前（xmin,ymin）左上的点对应旋转之后的值

    polys = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    polys = polys.reshape(-1, 4, 2)  # to (n, 4, 2) 第二维顺序：右上->右下->左下->左上

    return polys
```

### 计算两组rectangle所在直线的交点，并判断交点是否在两个端点之间

两点法直线簇的表示形式
<image src="../images/matheq/Two_line_segments_line.svg">

计算直线相交时的系数
<image src="../images/matheq/Two_line_segments_1.svg">

<image src="../images/matheq/Two_line_segments_2.svg">

若直线交点在原直线两端点之间有 $0\le t\le 1$ , $0\le u\le 1$ . 

计算各组直线的交点坐标
<image src="../images/matheq/Twoline_intersection.svg"> or <image src="../images/matheq/Twoline_intersection_1.svg">

```python
def get_intersection_points(polys1: torch.Tensor, polys2: torch.Tensor):
    """Find intersection points of rectangles
    Caculate the intersection point among each 4 line in a pair two rectangles, 
    judge whether the intersection on boundary of the pair rectangles.
    Convention: if two edges are collinear, there is no intersection point

    Args:
        polys1 (torch.Tensor): n, 4, 2
        polys2 (torch.Tensor): n, 4, 2

    Returns:
        intersectons (torch.Tensor): n, 4, 4, 2
        mask (torch.Tensor) : n, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([polys1, polys1[..., [1, 2, 3, 0], :]], 
                      dim=2)  # n, 4, 4: Box, edge, point 循环移位一位组成边
    line2 = torch.cat([polys2, polys2[..., [1, 2, 3, 0], :]], dim=2)
    # duplicate data to pair each edges from the boxes
    # (n, 4, 4) -> (n, 4, 4, 4) : Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(2).repeat([1, 1, 4, 1])
    line2_ext = line2.unsqueeze(1).repeat([1, 4, 1, 1])
    x1 = line1_ext[..., 0] #ploy1 x in colum repeat 4
    y1 = line1_ext[..., 1] #ploy1 y in colum repeat 4
    x2 = line1_ext[..., 2] #ploy1 x roll 1 in colum repeat 4
    y2 = line1_ext[..., 3] #ploy1 y roll 1 in colum repeat 4
    x3 = line2_ext[..., 0] #ploy2 x in line repeat 4
    y3 = line2_ext[..., 1] #ploy2 y in line repeat 4
    x4 = line2_ext[..., 2] #ploy2 x roll 1 in line repeat 4
    y4 = line2_ext[..., 3] #ploy2 y roll 1 in line repeat 4
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4) # 计算第一组的交点系数
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1，判断交点在两端点之间
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3) # 计算第二组的交点系数
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2, 判断交点在两端点之间
    mask = mask_t * mask_u                    # mask 表示两组直线交点在原polygon的边界上
    # overwrite with EPSILON. otherwise numerically unstable
    t = den_t / (num + EPSILON)
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)  # 计算两组直线的交点坐标
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask
```

### 判断rectangle的顶点是否在另一rectangle内部

<image src="../images/apex_in_rectangle.png">

```python
def get_in_box_points(polys1: torch.Tensor, polys2: torch.Tensor):
    """check if corners of poly1 lie in poly2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)

    Returns:
        c1_in_2: (n, 4) Bool
    """
    a = polys2[..., 0:1, :]  # (n, 1, 2) rectangle2 最右点
    b = polys2[..., 1:2, :]  # (n, 1, 2) rectangle2 最下点
    d = polys2[..., 3:4, :]  # (n, 1, 2) rectangle2 最上点
    ab = b - a                # (n, 1, 2)
    am = polys1 - a           # (n, 4, 2) rectangle1 到 rectangle2 最右点向量
    ad = d - a                # (n, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (n, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (n, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (n, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (n, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * \ 
        (p_ab / norm_ab < 1 + 1e-6)   # (n, 4) 
    cond2 = (p_ad / norm_ad > - 1e-6) * \
        (p_ad / norm_ad < 1 + 1e-6)   # (n, 4)
    #cond = ap*ab(ad)/ab(ad)^2 = |pa|*|ab|*cos(p_i,a,b)/(|ab|*|ab|)
    #判断p_i是否在rectangle内部，首先排除角p_i,a,b为钝角的点，此时cond<0,
    #若角p_i,a,b为锐角cond>0,若p落在rectangle边上有|ab|/|pa| = cos(p_i,a,b),此时cond=1，
    #p_i在rectangle内部|ab|/|pa| > cos(p_i,a,b),此时cond<1
    #若点p_i在renctangle外cond>1,在内则cond<=1
    #当满足cond=True对应的顶点，计算piou的相交面积时，顶点坐标作为intersection的顶点参与计算。
    return cond1 * cond2
```

### 拼接直线交点和box顶点，以及二者的mask
```python
def build_vertices(polys1: torch.Tensor, polys2: torch.Tensor,
                   c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor):
    """find vertices of intersection area

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (n, 4)
        c2_in_1 (torch.Tensor): Bool, (n, 4)
        inters (torch.Tensor): (n, 4, 4, 2)
        mask_inter (torch.Tensor): (n, 4, 4)

    Returns:
        两个rectangle,直线交点个数4x4,顶点个数8，输出维度4x4+8=24
        vertices (torch.Tensor): (n, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (n, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    n = polys1.size(0)
    # (n, 4+4+16, 2)
    vertices = torch.cat([polys1, polys2, inters.view(
        [n, -1, 2])], dim=1)
    # Bool (n, 4+4+16)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([n, -1])], dim=1) 
    # c1_in_2对应polys1是否是相交区域顶点
    # c2_in_1对应polys2是否是相交区域顶点
    # mask_inter.view([n, -1])对应各边所在直线的交点是否是相交区域顶点
    return vertices, mask  #对各焦点和顶点进行拼接，同时拼接各点是否是intersection的顶点
```

### 筛选相交区域的边界点，构成凸多边形，并且按相对中心点的顺序统一排列
```python
def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (n, 24, 2)
        mask (torch.Tensor): bool (n, 24)

    Returns:
        sorted_index: bool (n, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
        点数最多的情况每条边上各有两个焦点，重新补上一个起始点构成路线循环
    """
    # here we pad dim 0 to be consistent with the `sort_vertices_forward` function
    vertices = vertices.unsqueeze(0)
    mask = mask.unsqueeze(0)

    num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2,
                     keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)   #求相交区域多边形的重心坐标，

    # normalization makes sorting easier
    vertices_normalized = vertices - mean    #将坐标系变换到以相交区域重心为原点
    return sort_vertices_forward(vertices_normalized, mask, num_valid).squeeze(0).long()   
    #sort_vertices_forward 是c_cuda函数，作用大致是，从筛选出相交区域边界点，按顺序排布依照顺时针（逆时针），补初始点，其余值补0，可以用torch函数代替
```

### 计算相交区域的凸多边形面积

计算三角形面积如下图所示：
<image src="../images/triangle_area.jpeg" width = "400">

<image src="../images/matheq/triangle_area.svg">

推广到凸多边形
<image src="../images/convex_polygon_area.png">

<image src="../images/matheq/convex_polygon_area.svg"> 

依此类推 $S_{OABCD}$ , $S_{OABCDE}$ ...

```python
def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (n, 9)
        vertices (torch.Tensor): (n, 24, 2)

    return:
        area: (n), area of intersection
        selected: (n, 9, 2), vertices of polygon with zero padding 
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 2])
    selected = torch.gather(vertices, 1, idx_ext)  #select idx num in vertices
    total = selected[..., 0:-1, 0]*selected[..., 1:, 1] - \
        selected[..., 0:-1, 1]*selected[..., 1:, 0] 
    #以一个box的顶点各行为例x_0*y_1+x_1*y_2+...+x_n*y_0 - x_1*y_0 - x_2*y_1 -...-x_0*y_n
    total = torch.sum(total, dim=1)
    area = torch.abs(total) / 2
    return area, selected
```

### 对于任意两个ploygon计算其相交凸多边形的面积
```python
def oriented_box_intersection_2d(polys1: torch.Tensor, polys2: torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)

    Returns:
        area: (n,), area of intersection
        selected: (n, 9, 2), vertices of polygon with zero padding 
    """
    # find intersection points
    inters, mask_inter = get_intersection_points(polys1, polys2) #获得两个poly任意两两边的交点
    # find inter points
    c12 = get_in_box_points(polys1, polys2)                      #判断前者四个顶点是否在后者内部
    c21 = get_in_box_points(polys2, polys1)
    # build vertices
    vertices, mask = build_vertices(
        polys1, polys2, c12, c21, inters, mask_inter)
    # getting sorted indices
    sorted_indices = sort_indices(vertices, mask)
    # calculate areas using torch.gather
    return calculate_area(sorted_indices, vertices)
```

## 计算PIOU
<image src="../images/matheq/PIOU.svg">

```python
def box_iou_rotated_differentiable(boxes1: torch.Tensor, boxes2: torch.Tensor, iou_only: bool = True):
    """Calculate IoU between rotated boxes

    Args:
        box1 (torch.Tensor): (n, 5)
        box2 (torch.Tensor): (n, 5)
        iou_only: Whether to keep other vars, e.g., polys, unions. Default True to drop these vars.

    Returns:
        iou (torch.Tensor): (n, )
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        U (torch.Tensor): (n) area1 + area2 - inter_area
    """
    # transform to polygons
    polys1 = rotated_box_to_poly(boxes1)                          #将box转换成四角坐标
    polys2 = rotated_box_to_poly(boxes2)
    # calculate insection areas
    inter_area, _ = oriented_box_intersection_2d(polys1, polys2)  #计算交集面积
    area1 = boxes1[..., 2] * boxes1[..., 3]                       #w_1*h_1
    area2 = boxes2[..., 2] * boxes2[..., 3]                       #w_2*h_2
    union = area1 + area2 - inter_area                            #由交集求并集, w_1*h_1 + w_2*h_2 - intersection
    iou = inter_area / union                                      #iou = intersection/(w_1*h_1 + w_2*h_2 - intersection)
    if iou_only:
        return iou
    else:
        return iou, union, polys1, polys2,
```

## 计算PIOU-loss
<image src="../images/matheq/PIOU_loss.svg">

```python
def iou_loss(pred, target, linear=False, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = box_iou_rotated_differentiable(pred, target).clamp(min=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss
```

