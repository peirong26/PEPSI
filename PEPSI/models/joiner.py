

"""
Wrapper interface.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

from PEPSI.models.unet3d.model import UNet3D
from .head import IndepHead, MultiInputIndepHead
from utils.checkpoint import load_checkpoint 


#supersynth_ckp_path = 'ckp/wmh-synthseg/PAPER_checkpoint_0101.pth'
supersynth_ckp_path = 'ckp/wmh-synthseg/AllDataIn_checkpoint_0101.pth'

flair2pathol_feat_ckp_path = 'ckp/Supv/supv_adni3_flair2pathol_feat_epoch_35.pth' 
flair2pathol_task_ckp_path = 'ckp/Supv/supv_adni3_flair2pathol_epoch_35.pth' 



def build_supersynth_model(device = 'cpu'):
    # 33 + 4 + 1 + 1 = 39 (SuperSynth)
    backbone = UNet3D(1, f_maps=64, layer_order='gcl', num_groups=8, num_levels=5, is3d=True)
    head = IndepHead(None, f_maps_list = [64], out_channels ={'seg': 39}, is_3d = True, out_feat_level = -1)
    model = get_joiner('seg', backbone, head) 
    processor = SegProcessor().to(device)
    model.to(device)
    return model, processor


def build_pathol_model(device = 'cpu'):
    backbone = UNet3D(1, f_maps=64, layer_order='gcl', num_groups=8, num_levels=5, is3d=True)
    feat_model = get_joiner('seg', backbone, None) 
    task_model = MultiInputIndepHead(None, [64], {'pathol': 1}, True, -1)
    processor = PatholProcessor().to(device)
    feat_model.to(device)
    task_model.to(device)
    return feat_model, task_model, processor




class UncertaintyProcessor(nn.Module):
    def __init__(self, output_names):
        super(UncertaintyProcessor, self).__init__()
        self.output_names = output_names

    def forward(self, outputs, *kwargs): 
        for output_name in self.output_names:
            if 'image' in output_name:
                for output in outputs:
                    output[output_name + '_sigma'] = output[output_name][:, 1][:, None]
                    output[output_name] = output[output_name][:, 0][:, None]
        return outputs

class SegProcessor(nn.Module):
    def __init__(self):
        super(SegProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['seg'] = self.softmax(output['seg'])
        return outputs
    
class PatholProcessor(nn.Module):
    def __init__(self):
        super(PatholProcessor, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['pathol'] = self.sigmoid(output['pathol'])
        return outputs
 
class PatholSeg(nn.Module):
    def __init__(self, args):
        super(PatholSeg, self).__init__()
        self.sigmoid = nn.Sigmoid()

        paths = args.supervised_pathol_seg_ckp_path
        self.feat_model, self.task_model, self.processor = build_pathol_model()
        load_checkpoint(paths.feat, [self.feat_model], model_keys = ['model'], to_print = False)
        load_checkpoint(paths.task, [self.task_model], model_keys = ['model'], to_print = False)
        for param in self.feat_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
            param.requires_grad = False
        for param in self.task_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
            param.requires_grad = False

        aux_paths = args.supervised_aux_pathol_seg_ckp_path
        if args.aux_modality is not None:
            self.aux_feat_model, self.aux_task_model, self.aux_processor = build_pathol_model()
            load_checkpoint(aux_paths.feat, [self.aux_feat_model], model_keys = ['model'], to_print = False)
            load_checkpoint(aux_paths.task, [self.aux_task_model], model_keys = ['model'], to_print = False)
            for param in self.aux_feat_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
                param.requires_grad = False
            for param in self.aux_task_model.parameters():  # Crucial!!!! We backprop through it, but weights should not change
                param.requires_grad = False
        else:
            self.aux_feat_model, self.aux_task_model, self.aux_processor = None, None, None

    def forward(self, outputs, subjects, curr_dataset, *kwargs): 
        for output in outputs:
            if output['image'].shape == subjects['image'].shape:
                samples = [ { 'input': output['image'] },  { 'input': subjects['image'] } ]
                feats, inputs = self.feat_model(samples)
                preds = self.task_model([feat['feat'] for feat in feats], inputs)
                preds = self.processor(preds, samples)
                output['implicit_pathol_pred'] = preds[0]['pathol']
                output['implicit_pathol_orig'] = preds[1]['pathol'] 
            if self.aux_feat_model is not None:
                if output['aux_image'].shape == subjects['aux_image'].shape:
                    samples = [ { 'input': output['aux_image'] }, { 'input': subjects['aux_image'] } ]
                    feats, inputs = self.aux_feat_model(samples)
                    preds = self.aux_task_model([feat['feat'] for feat in feats], inputs)
                    preds = self.processor(preds, samples)
                    output['implicit_aux_pathol_pred'] = preds[0]['pathol']
                    output['implicit_aux_pathol_orig'] = preds[1]['pathol'] 
        return outputs
    
class ContrastiveProcessor(nn.Module):
    def __init__(self):
        '''
        Ref: https://openreview.net/forum?id=2oCb0q5TA4Y
        '''
        super(ContrastiveProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output['feat'][-1] = F.normalize(output['feat'][-1], dim = 1)
        return outputs

class BFProcessor(nn.Module):
    def __init__(self):
        super(BFProcessor, self).__init__()

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['bias_field'] = torch.exp(output['bias_field_log'])
        return outputs


##############################################################################


class MultiInputIndepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head):
        super(MultiInputIndepJoiner, self).__init__()

        self.backbone = backbone 
        self.head = head

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat = self.backbone.get_feature(x['input'])
            out = {'feat': feat}
            if self.head is not None: 
                out.update( self.head(feat) )
            outs.append(out)
        return outs, [input['input'] for input in input_list]


class MultiInputDepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head):
        super(MultiInputDepJoiner, self).__init__()

        self.backbone = backbone 
        self.head = head

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat = self.backbone.get_feature(x['input'])
            out = {'feat': feat} 
            if self.head is not None: 
                out.update( self.head( feat, x['input']) )
            outs.append(out)
        return outs, [input['input'] for input in input_list]



################################


def get_processors(args, task, device):
    processors = []
    if args.losses.uncertainty is not None:
        processors.append(UncertaintyProcessor(args.output_names).to(device))
    if 'contrastive' in task:
        processors.append(ContrastiveProcessor().to(device))
    if 'seg' in task:
        processors.append(SegProcessor().to(device))
    if 'pathol' in task:
        processors.append(PatholProcessor().to(device))
    if args.losses.implicit_pathol: 
        processors.append(PatholSeg(args).to(device))
    if 'bf' in task:
        processors.append(BFProcessor().to(device))
    return processors


def get_joiner(task, backbone, head):
    if 'sr' in task or 'bf' in task:
        return MultiInputDepJoiner(backbone, head) 
    else:
        return MultiInputIndepJoiner(backbone, head)