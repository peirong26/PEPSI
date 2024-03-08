##############################
#######   PEPSI Demo   #######
##############################


import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.demo_utils import prepare_image, evaluate_image
from utils.misc import viewVolume, make_dir
 

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


ckp_path = 'assets/pepsi_pretrained.pth'

imgs = {
    'T1w': 'assets/T1w.nii.gz',
    'FLAIR': 'assets/FLAIR.nii.gz',
    }

gold_standard_pathologies = {
    'T1w': 'assets/T1w_Gold_Standard_Pathology.nii.gz',
    'FLAIR': 'assets/FLAIR_Gold_Standard_Pathology.nii.gz',
}



input_modality = 'T1w' # Try different input images with different pathologies!
num_plot_feats = 1 # 64 features from the last layer in total




img_path = imgs[input_modality]


im, aff = prepare_image(img_path, device = device)
outputs = evaluate_image(im, ckp_path, feature_only = False, device = device)


# Check outputs
for k in outputs.keys(): 
    print('out:', k)


# Get PEPSI synthesized MP-RAGE (Anatomy Image with Darkened Anomalies)
mprage = outputs['image']
print(mprage.size()) # (1, 1, h, w, d)
viewVolume(mprage, aff, names = ['out_mprage_from_%s' % input_modality], save_dir = make_dir('outs'))

# Get PEPSI synthesized FLAIR (Pathology Image with Brightened Anomalies)
flair = outputs['aux_image']
print(flair.size()) # (1, 1, h, w, d)
viewVolume(flair, aff, names = ['out_flair_from_%s' % input_modality], save_dir = make_dir('outs'))


# Get PEPSI features
feats = outputs['feat'][-1]
print(feats.size()) # (1, 64, h, w, d)
# Uncomment the following if you want to save the features
# NOTE: feature size could be large
for i in range(num_plot_feats): 
  viewVolume(feats[:, i], aff, names = ['feat-%d' % (i+1)], save_dir = make_dir('outs/feats-%s' % input_modality))
