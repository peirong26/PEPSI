"""
Backbone modules.
"""

import torch.nn as nn 

from PEPSI.models.unet3d.model import UNet3D


backbone_options = {
    'unet3d': UNet3D,
}



####################################


def build_backbone(args):

    backbone = backbone_options[args.backbone](args.in_channels, args.f_maps, 
                                               args.layer_order, args.num_groups, args.num_levels, 
                                               args.unit_feat,
                                               )
    
    return backbone

