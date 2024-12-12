# Object Detection 

## Detection Transformer (DETR)

Paper:[End-to-end object detection with transformers](https://arxiv.org/pdf/2005.12872v3)

<image src="./images/DETR.png">

Detection Transformer 将由backbone获得的feature token作为input embedding送入Transformer Encoder-Decoder结构中，作为output embedding的object queries的输出进入shared FFN来估计目标的bbox和cls,或者“no object”的token,预测的目标token和实际目标之间采用匈牙利匹配建立联系。因为类似的结构已经在STARK(visual object track)这类任务中，所以这里不再给出结构代码。

The code comes from https://github.com/facebookresearch/detr

Detr采用匈牙利法来匹配模型的object query的输出和tgt_box,并且计算多个阶段的decoder输出的loss,作为辅助loss来提升网络性能。
<details>

```python
# comes from /models/detr.py
class SetCriterion(nn.Module):
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 在detr中采用多个阶段的decoder query的output token的loss作为辅助损失
        # 辅助损失可以帮助网络加快收敛和维持Rubust性能，并提升准确率。在visual object tracking中单一阶段的估计结果不好也证明了这点
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
```
</details>

匈牙利法匹配目标和输出之间的过程如下：
<details>

```python
# comes from /models/matcher.py
class HungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] #筛选每个目标class对应每个output的概率，组成矩阵
        # cost_class：[num_queries len(targets)] 每个目标分属于每个tgt对应类型的概率

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) #计算out和traget二者中的每个元素间的p范数，越接近越小
        # out_bbox [batch_size, num_queries, 4]，tgt_bbox [batch_size, len(target), 4] cost_bbox [batch_size num_queries  len(target)]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # cost_giou [batch_size num_queries  len(target)]
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # 加权值，越接近越小
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # scipy.optimize.linear_sum_assignment 使用匈牙利匹配返回任务分配最小化总成本时的索引(即tgt中的目标分配给哪个output)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
```
</details>

## Grounfed Lamguage-Image Pretraning (GLIP)

Paper:[Grounfed lamguage-image pretraning](https://arxiv.org/pdf/2112.03857v2)

<image src="./images/GLIP.png">

<details>
<summary>在介绍GLIP之前需要先了解CLIP</summary>

### Contrastive Lauguage-Image Pretraining (CLIP)

Paper:[Learning transferable visual models from natural langeage supervision](https://arxiv.org/pdf/2103.00020v1)

<image src="./images/CLIP.jpg">

CLIP网络使用Text Encoder和Image Encoder两个模块分别提取文本和图像特征，优化二者的余弦相似性获得文本-图像的匹配关系。通过使用了自然语言监督，和one-hot 
标签训练的网络相比，CLIP可以更充分的利用目标状态空间，打破了固定类别标签，获得了更加灵活的迁移能力，同时减少网络训练对于标注数据的依赖。通过对比学习CLIP可以获得同一组数据中的不同样本之间的相似度和差异性。

```python

```

</details>

Once I thought GLIP would be a two-stage detection, rerange the patchs output by the image encoder as the feature, get the box regression predict from RPN, then by Roi Align generate the fixed size patchs feature, do the Contrastive lauguage-Image loss in the roi heads. But after get throught the microsoft codes of GLIP, the network is performan as RPN only architecture, which is same as the one-stage detection as YOLO did. After the RPN generate the cls and box regression for each anchors and select the positive anchor by ATSS, it use the token of image patch, to generate loss between language and image, and direct pridictor the regression loss of box. 

The code comes from https://github.com/microsoft/GLIP

GLIP的Text和Image之间的fusion,参考公式如下

$
O^i_q =O^iW_{q,Im},P^i_q =P^iW_{q,Lan},Attn =O^i_q(P^i_q)^T\sqrt{d} \\
O^i_v =O^iW_{v,Im}, O^i_{t2i} =SoftMax(Attn)P^i_vW_{out,Im} \\
P^i_v =P^iW_{v,Lan},P^i_{i2t} =SoftMax(Attn^T)O^i_vW_{out,Lan} \\
O^{i+1} =DyHead(O^i+O^i_{t2i}) \\
P^{i+1} =BERTLayer(P^i+P^i_{i2t}) \\
where i \in (0,1,\cdots,L-1)
$

其在代码中的实现过程如下：

计算$O^i_{t2i}$,$P^i_{i2t}$
<details>

```python
# comes from /maskrcnn_benchmark/utils/fuse_helper.py
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim) # image embed as q,k
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim) # langueage embed as q,k
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim) # value embed of image
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim) # value embed of langueage 

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim) # output embed of image
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim) # output embed of langueage

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size() # tgt_len：feature map lens, HxW

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        # when doing with 3 dim matrix, torch.bmm is same as torhc.matmul, and faster
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])

        attn_weights = torch.clamp(attn_weights_l, min=-50000, max=50000)
        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            # 对language token添加mask
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l
```
</details>
计算O^{i+1}，P^{i+1}
<details>

```python
class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim,
            num_heads=num_heads, dropout=dropout, cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            visu_feat = []
            lang_feat = []
            # 因为GLIP允许输入的每张图的尺寸不同，使用transformer做image encoder图像被当作序列，不在受长宽限制，
            # 这里以单张图的形式计算Text2Image和Image2Text的多头注意力
            for ii, feat in enumerate([q0, q1, q2, q3, q4]):
                bs, _, h, w = feat.shape
                q = feat.flatten(2).transpose(1, 2)
                
                new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
                new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
                lang_feat.append(new_l)
                visu_feat.append(new_v)
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                pass
            else:
                lang_feat = self.shrink_lang(torch.cat(lang_feat, dim = -1)) # From multiple dimensions
                lang_feat = [lang_feat, None, None, None, None]
        else:
            # 通过将所有的image model output连接到一起(batch and stage)，一次性计算attn
            visu_feat = []
            size_per_level, visual_features_flatten = [], []
            for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
                bs, c, h, w = feat_per_level.shape
                size_per_level.append([h, w]) #记录送入时的序列长度
                # permute_and_flatten (N, A, C, H, W) -> (N, (A H W), C)
                feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                visual_features_flatten.append(feat)
            visual_features_flatten = cat(visual_features_flatten, dim=1) # visual concatenate到一起做attn,做完后在切开
            # 因为VLfuse是在word patch和image patch之间，对确定的分类任务language token一致所以可以这么做
            new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l)
            # [bs, N, C] -> [bs, C, N]
            new_v = new_v.transpose(1, 2).contiguous()

            start = 0
            for (h, w) in size_per_level:
                # 根据记录的长度还原每张图片的Text2Image output feature
                new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
                visu_feat.append(new_v_per_level)
                start += h * w
            
            lang_feat = [new_l, None, None, None, None] # 在上级函数VLFuse中只保留lang_feat[0]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3], lang_feat[4]

    
    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l
```
</details>
VLDyHead is the rpn archtecture here
<details>

```python
class VLDyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLDyHead, self).__init__()
        self.cfg = cfg
        lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)

        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES - 1
        num_tokens = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE # 长宽比x面积数
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        if cfg.MODEL.DYHEAD.USE_GN: # ? batch normalize type
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU # here True
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE # here True
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV # here True

        conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            # 因为VLFuse的output 是字典的形式{"visual": fused_visual_features,"lang": fused_language_dict_features}
            # 在BertLayer和DyConv中源码通过选择key来控制输入
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:
                # cross-modality fusion
                dyhead_tower.append(VLFuse(cfg))
                # BertLayer get language stage output
                dyhead_tower.append(
                    BertEncoderLayer(lang_cfg,
                            clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                            clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                )

            # self vision path
            dyhead_tower.append(
                # DyConv Deformable offset conv get image stage output
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE

        # dot product soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS == False
            self.dot_product_projection_image = nn.Identity()
            self.dot_product_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                         num_anchors * channels, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            # DEBUG
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # if use dot product token loss
        for modules in [self.dot_product_projection_image]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, bias_value)

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None):
        logits,bbox_reg,centerness = [],[],[]

        feat_inputs = {"visual": x,"lang": language_dict_features}
        # dyhead_tower的输出{"visual": fused_visual_features,"lang": fused_language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        t_logits,contrastive_logits,proj_tokens,mlm_logits = None,None,None,None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
            embedding = dyhead_tower["lang"]["hidden"]
        
        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            dot_product_logits = []
            # norm
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
            # w/o norm
            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats,shallow_text_emb = None,None

        fused_visual_features = None
        if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
            fused_visual_features = []

        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l])) # logits class token

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred) # bbox_reg box token

            centerness.append(self.centerness(dyhead_tower["visual"][l])) 
            # centerness mean how much the box belong to the anchor center

            x = dyhead_tower["visual"][l]
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                fused_visual_features.append(x)
            B, C, H, W = x.shape

            # add bias (language)
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            # dot_product_proj_queries: token from image; dot_product_proj_tokens token from language 
            # cosine distance between image token and language token
            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000， max=50000)
            dot_product_logits.append(dot_product_logit)

        return logits, bbox_reg, centerness, 
        t_logits, proj_tokens, contrastive_logits, #None
        dot_product_logits, # cosine distance
        mlm_logits, shallow_img_emb_feats, fused_visual_features #None
```
</details>
For training the output of VLDyHead, GLIP do as the same RPN ROI region select did use ATSStargetAlign, select the positive and the negative anchor toekn output by VLDyHeadModule rerange the matrix to caculate loss. The cosine distance of image and language is to compuate the TokenSigmoidFocalloss.

When inferece the cosine distance used to repalce the cls score for box genereted.

# 内容延伸

1. [Transforner Base Architecture](./README_transformer.md)

2. [TR Encoder Model ViTs](./README_ViT.md)

3. [Pos-Emb in TR](./README_pos_emb.md)

4. [Visual object track](./README_visual_object_track.md)

5. [Object Detection based on transformer](./README_detection.md)