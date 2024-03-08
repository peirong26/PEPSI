

"""
Datasets interface.
"""
from .synth import BaseSynth
from .id_synth import IDSynth, DeformIDSynth
from .id_synth_eval import IDSynthEval, DeformIDSynthEval
from .supervised import ContrastSpecificDataset


dataset_options = { 
    'train':{
        'synth': BaseSynth,  
        'synth_id': IDSynth,  
        'synth_id_deform': DeformIDSynth,
        'supervised': ContrastSpecificDataset,
    },
    'test':{
        'synth_id': IDSynthEval,  
        'synth_id_deform': DeformIDSynthEval,
        'supervised': ContrastSpecificDataset,
    }
}


dataset_paths = {
    'synth-WMH': {
        'type': 'wmh',
        'train': 'data/synthseg',
    },
    'synth-ADNI': {
        'type': 'wmh',
        'train': 'data/synth',
    },
    'synth-ADNI_crop': { # with synthesized FLAIR
        'type': 'wmh',
        'train': 'data/synth_crop',
    },
    'synth-ADNI3': { # wm lesion
        'type': 'wmh',
        'train': 'data/adni3',
    },
    'synth-BraTS': { # brain tumor
        'type': 'tumor',
        'train': 'data/brats',
    },
    'synth-ATLAS': { # stroke
        'type': 'stroke',
        'train': 'data/atlas',
    },
    'synth-ISLES': { # stroke
        'type': 'stroke',
        'train': 'data/isles2022',
    },
    'ADNI': { # wm lesion
        'type': 'wmh',
        'T1': 'data/synth/T1',
        'Seg': 'data/synth/label_maps_segmentation',
        'Pathol': 'data/synth/pathology_maps_segmentation',
    },
    'ADNI-synthflair': { # wm lesion, FLAIR is fake, generated from Supv T1->FLAIR trained on ADNI3
        'type': 'wmh',
        'T1': 'data/synth_crop/T1',
        'FLAIR': 'data/synth_crop/FLAIR',
        'Seg': 'data/synth_crop/label_maps_segmentation',
        'Pathol': 'data/synth_crop/pathology_maps_segmentation',
    },
    'ADNI3': { # wm lesion
        'type': 'wmh',
        'T1': 'data/adni3/T1', # T1toFLAIR'
        'FLAIR': 'data/adni3/FLAIR',
        'Seg': 'data/adni3/label_maps_segmentation', # T1toFLAIR-synthseg'
        'Pathol': 'data/adni3/pathology_maps_segmentation',
        'PatholProb': 'data/adni3/pathology_probability',
    },
    'ADHD200': { # ADHD
        'type': 'adhd',
        'T1': 'ADHD200/T1',
        'Seg': 'ADHD200/T1-SynthSeg',
    },
    'AIBL': { # wm lesion
        'type': 'wmh',
        'T1': 'AIBL/T1',
        'T2': 'AIBL/T2',
        'FLAIR': 'AIBL/FLAIR',
        'Seg': 'AIBL/T1-SynthSeg',
    },
    'HCP': { # healthy
        'type': None,
        'T1': 'HCP/T1',
        'T2': 'HCP/T2',
        'Seg': 'HCP/T1-SynthSeg',
    },
    'OASIS3': {
        'type': None,
        'CT': 'OASIS3/CT',
        'T1': 'OASIS3/T1toCT',
        'Seg': 'OASIS3/T1toCT-SynthSeg',
    }, 
    'BraTS': { # brain tumor
        'type': 'tumor',
        'T1': 'data/brats/T1',
        'FLAIR': 'data/brats/FLAIR',
        'Pathol': 'data/brats/pathology_maps_segmentation',
        'Seg': 'data/brats/label_maps_segmentation', 
    },
    'ATLAS': { # stroke
        'type': 'stroke',
        'T1': 'data/atlas/T1',
        'Pathol': 'data/atlas/pathology_maps_segmentation',
        'Seg': 'data/atlas/label_maps_segmentation', 
    },
    'ISLES': { # stroke
        'type': 'stroke',
        'T1': 'data/isles2022/T1',
        'FLAIR': 'data/isles2022/FLAIR',
        'Pathol': 'data/isles2022/pathology_maps_segmentation',
        'Seg': 'data/isles2022/label_maps_segmentation', 
    },
}


def get_dir(dataset, modality_input, modality_target, aux_modality, split, task): 
    if 'synth' in dataset: 
        return dataset_paths[dataset][split], None, None
    else: 
        aux_dir = None if aux_modality is None else dataset_paths[dataset][aux_modality]
        if 'seg' in task or 'bf' in task:
            return dataset_paths[dataset][modality_input], dataset_paths[dataset]['Seg'], aux_dir
        elif 'pathol' in task:
            return dataset_paths[dataset][modality_input], dataset_paths[dataset]['Pathol'], aux_dir
        elif 'anat' in task: 
            return dataset_paths[dataset][modality_input], dataset_paths[dataset][modality_target], aux_dir
        elif 'sr' in task:
            return dataset_paths[dataset][modality_input], dataset_paths[dataset][modality_input], aux_dir
        else:
            raise ValueError('Unsupported task type:', task)




def build_dataset_single(dataset_name, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    data_dir, gt_dir, aux_gt_dir = get_dir(args.dataset, args.modality_input, args.modality_target, args.aux_modality, split, args.task)
    if 'supervised' in dataset_name:
        dataset = dataset_options[split][dataset_name](args, data_dir, gt_dir, aux_gt_dir, device)
    else:
        dataset = dataset_options[split][dataset_name](args, data_dir, device)
    dataset.name = args.dataset
    return dataset


def build_dataset_multi(dataset_name, split, args, device):
    """Helper function to build dataset for different splits ('train' or 'test')."""
    datasets = {}
    for name in args.dataset:
        data_dir, gt_dir, aux_gt_dir = get_dir(name, args.modality_input, args.modality_target, args.aux_modality, split, args.task)
        if 'supervised' in dataset_name:
            curr_dataset = dataset_options[split][dataset_name](args, data_dir, gt_dir, aux_gt_dir, device)
        else:
            curr_dataset = dataset_options[split][dataset_name](args, data_dir, device)
        curr_dataset.name = name
        datasets[name] = curr_dataset
    return datasets


def build_dataset(datasets, split, args, device):
    if isinstance(args.dataset, str):
        return {args.dataset: build_dataset_single(datasets, split, args, device)}
    else:
        assert isinstance(args.dataset, list)
        return build_dataset_multi(datasets, split, args, device)
