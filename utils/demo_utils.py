import os
import numpy as np
import torch

from PEPSI.datasets.utils import fast_3D_interp_torch
from PEPSI.models import build_feat_model, build_downstream_model
from utils.checkpoint import load_checkpoint 
import utils.misc as utils 



# default & gpu cfg #
default_cfg_file = 'cfgs/default_train.yaml'
default_data_file = 'cfgs/default_dataset.yaml'
submit_cfg_file = 'cfgs/submit.yaml'


def center_crop(img, win_size = [220, 220, 220]):
    # center crop
    if len(img.shape) == 4: 
        img = torch.permute(img, (3, 0, 1, 2)) # (move last dim to first)
        img = img[None]
        permuted = True
    else: 
        assert len(img.shape) == 3
        img = img[None, None]
        permuted = False

    orig_shp = img.shape[2:] # (1, d, s, r, c)
    if win_size is None:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp
    elif orig_shp[0] > win_size[0] or orig_shp[1] > win_size[1] or orig_shp[2] > win_size[2]:
        crop_start = [ max((orig_shp[i] - win_size[i]), 0) // 2 for i in range(3) ]
        crop_img = img[ :, :, crop_start[0] : crop_start[0] + win_size[0], 
                   crop_start[1] : crop_start[1] + win_size[1], 
                   crop_start[2] : crop_start[2] + win_size[2]]
        if permuted:
            return torch.permute(crop_img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return crop_img, crop_start, orig_shp
    else:
        if permuted:
            return torch.permute(img, (0, 2, 3, 4, 1)), [0, 0, 0], orig_shp
        return img, [0, 0, 0], orig_shp


def prepare_image(img_path, win_size = [220, 220, 220], spacing = None, is_label = False, im_only = False, device = 'cpu'):
    im, aff = utils.MRIread(img_path, im_only=False, dtype='int' if is_label else 'float')
    im = torch.tensor(np.squeeze(im), dtype=torch.int if is_label else torch.float32, device=device)
    im = torch.nan_to_num(im)
    if not is_label:
        im -= torch.min(im)
        im /= torch.max(im)
    im, aff = utils.torch_resize(im, aff, 1.)
    if spacing is not None:
        im = resample(im, new_res = spacing)
    im, aff = utils.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    im, crop_start, orig_shp = center_crop(im, win_size)
    if im_only:
        return im
    return im, aff


def resample(I, orig_res = [1., 1., 1.], new_res = [1., 1., 1.]):
    if not isinstance(orig_res, list):
        orig_res = [orig_res, orig_res, orig_res]
    if not isinstance(new_res, list):
        new_res = [new_res, new_res, new_res]
    #print('pre resample', I.shape)
    resolution = np.array(new_res)
    new_size = (np.array(I.shape) * orig_res / resolution).astype(int)

    factors = np.array(new_size) / np.array(I.shape)
    delta = (1.0 - factors) / (2.0 * factors)
    vx = np.arange(delta[0], delta[0] + new_size[0] / factors[0], 1 / factors[0])[:new_size[0]]
    vy = np.arange(delta[1], delta[1] + new_size[1] / factors[1], 1 / factors[1])[:new_size[1]]
    vz = np.arange(delta[2], delta[2] + new_size[2] / factors[2], 1 / factors[2])[:new_size[2]]
    II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
    II = torch.tensor(II, dtype=torch.float, device=I.device)
    JJ = torch.tensor(JJ, dtype=torch.float, device=I.device)
    KK = torch.tensor(KK, dtype=torch.float, device=I.device)

    I_resize = fast_3D_interp_torch(I, II, JJ, KK, 'linear') 
    I_new = utils.myzoom_torch(I_resize, 1 / factors) 

    #print('post resample', I_new.shape)
    return I_new


@torch.no_grad()
def evaluate_image(inputs, feat_ckp_path, task_ckp_path = None, task = 'uni-feat-anat-flair-aux', feature_only = True, device = 'cpu'):
    # inputs: Torch.Tensor -- (batch_size, 1, s, r, c)

    # ============ prepare ... ============
    args = utils.preprocess_cfg([default_cfg_file, default_data_file, submit_cfg_file])
    args.task = task
    args.unit_feat = 'uni' in task
    samples = [ { 'input': inputs } ]

    # ============ testing ... ============
    if task_ckp_path is None:
        args, feat_model, processors, criterion, postprocessor = build_feat_model(args, device = device)
        load_checkpoint(feat_ckp_path, [feat_model], model_keys = ['model'], to_print = False)
        outputs, _ = feat_model(samples) # dict with features
    else:
        args, feat_model, task_model, processors, criterion, postprocessor = build_downstream_model(args, device = device)  
        load_checkpoint(feat_ckp_path, [feat_model], model_keys = ['model'], to_print = False)
        load_checkpoint(task_ckp_path, [task_model], model_keys = ['model'], to_print = False) 
        feats, inputs = feat_model(samples)
        outputs = task_model([feat['feat'] for feat in feats], inputs)
        outputs[0]['feat'] = feats[0]['feat']

    for processor in processors:
        outputs = processor(outputs, samples)
    if postprocessor is not None:
        outputs = postprocessor(args, outputs)

    if feature_only:
        return outputs[0]['feat'][-1] # (batch_size, 64, s, r, c)
    else:
        return outputs[0]



@torch.no_grad()
def evaluate_path(input_paths, save_dir, feat_ckp_path, task_ckp_path = None, task = 'feat', win_size = [220, 220, 220], 
                  save_input = False, aux_paths = {}, save_aux = False, exclude_keys = [], mask_output = False, ext = '.nii.gz', device = 'cpu'):
     
    args = utils.preprocess_cfg([default_cfg_file, default_data_file, submit_cfg_file])
    args.task = task
    mask = None

    # ============ loading ... ============
    if task_ckp_path is None:
        args, feat_model, processors, criterion, postprocessor = build_feat_model(args, device = device)
        load_checkpoint(feat_ckp_path, [feat_model], model_keys = ['model'], to_print = False)
    else:
        args, feat_model, task_model, processors, criterion, postprocessor = build_downstream_model(args, device = device)  
        load_checkpoint(feat_ckp_path, [feat_model], model_keys = ['model'], to_print = False)
        load_checkpoint(task_ckp_path, [task_model], model_keys = ['model'], to_print = False) 

    for i, input_path in enumerate(input_paths):

        print('Now testing: %s (%d/%d)' % (input_path, i+1, len(input_paths)))
        curr_save_dir = utils.make_dir(os.path.join(save_dir, os.path.basename(input_path).split('.nii')[0]))

        # ============ prepare ... ============
        im, aff, crop_start, orig_shp = prepare_image(input_path, win_size, device = device)
        if save_input:
            print('  Input: saved in - %s' % (os.path.join(curr_save_dir, 'input' + ext)))
            utils.viewVolume(im, aff, names = ['input'], ext = ext, save_dir = curr_save_dir)
        for k in aux_paths.keys():
            im_k, _, _, _ = prepare_image(aux_paths[k][i], win_size, is_label = 'label' in k, device = device)
            if save_aux:
                print('  Aux input: %s - saved in - %s' % (k, os.path.join(curr_save_dir, k + ext)))
                utils.viewVolume(im_k, aff, names = [k], ext = ext, save_dir = curr_save_dir)
            if mask_output and 'mask' in k:
                mask = im_k.clone()
                mask[im_k != 0.] = 1.
        samples = [ { 'input': im } ]
    
        # ============ testing ... ============
        if task_ckp_path is None:
            outputs, _ = feat_model(samples) # dict with features
        else:
            feats, inputs = feat_model(samples)
            outputs = task_model([feat['feat'] for feat in feats], inputs)

        for processor in processors:
            outputs = processor(outputs, samples)
        if postprocessor is not None:
            outputs = postprocessor(args, outputs)

        out = outputs[0]
        if mask_output and mask is None:
            mask = torch.zeros_like(im)
            mask[im != 0.] = 1.
        for key in out.keys():
            if key not in exclude_keys and isinstance(out[key], torch.Tensor):
                print('  Output: %s - saved in - %s' % (key, os.path.join(curr_save_dir, 'out_' + key + ext)))
                out[key][out[key] < 0.] = 0.
                utils.viewVolume(out[key] * mask if mask_output else out[key], aff, names = ['out_'+key], ext = ext, save_dir = curr_save_dir)
