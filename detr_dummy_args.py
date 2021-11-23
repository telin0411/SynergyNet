import numpy as np


class dummy_args(object):
    def __init__(self):
        self.with_box_refine = False
        self.two_stage = False
        self.backbone = "resnet50"
        self.dilation = True
        self.position_embedding = "learned"
        self.position_embedding = "sine"
        self.position_embedding_scale = 2 * np.pi
        self.num_feature_levels = 4
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 1024
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 300
        self.dec_n_points = 4
        self.enc_n_points = 4
        self.masks = False
        self.no_aux_loss = False
        self.set_cost_class = 2
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.cls_loss_coef = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.focal_alpha = 0.25
        self.dataset_file = "coco"
        self.lr_backbone = 2e-5
        self.aux_loss = False
