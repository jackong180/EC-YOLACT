"""Shared experiment configuration and checkpoint validation."""
import json,random
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from nets.yolact import Yolact
ABLATIONS={'baseline':('none','none'),'a':('ega','none'),'b':('none','ega'),'c':('ca','none'),'d':('none','ca'),'ec':('ca','ega')}
def read_config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    if len(c['anchor_sizes'])!=5: raise ValueError('Five anchor sizes required')
    return c

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def seed_worker(_):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)

def make_model(c,variant,training=True):
    b,p=ABLATIONS[variant]
    return Yolact(len(c['classes'])+1,backbone=c['backbone'],backbone_attention=b,prototype_attention=p,
                  ca_normalization=c['ca_normalization'],train_mode=training)

def get_data(c,split):
    co=COCO(c[f'{split}_annotations']); byname={v['name']:k for k,v in co.cats.items()}
    if set(byname)!=set(c['classes']): raise ValueError('Dataset category names do not match config')
    return co,{byname[n]:i+1 for i,n in enumerate(c['classes'])}

def check_data(c,require_images=True):
    splits={}; report={}
    for split in ('train','val'):
        co,_=get_data(c,split)
        if not co.imgs: raise ValueError(f'{split}: empty dataset')
        for ann in co.anns.values():
            if ann['image_id'] not in co.imgs or ann['category_id'] not in co.cats: raise ValueError('Annotation references missing image/category')
            if ann['bbox'][2]<=0 or ann['bbox'][3]<=0 or ann.get('area',0)<=0 or not ann.get('segmentation'): raise ValueError(f'Invalid box/area/segmentation: {ann["id"]}')
        names=[a['file_name'] for a in co.imgs.values()]
        if len(names)!=len(set(names)): raise ValueError(f'Duplicate filenames in {split}')
        missing=[n for n in names if not (Path(c[f'{split}_images'])/n).is_file()]
        report[split]={'images':len(names),'annotations':len(co.anns),'missing_images':len(missing)}
        splits[split]=set(names)
        if require_images and missing: raise FileNotFoundError(f'{split}: {len(missing)} images missing; first: {missing[0]}')
    overlap=splits['train']&splits['val']; report['filename_overlap']=len(overlap)
    if overlap: raise ValueError('Train/validation filename overlap: data leakage')
    return report

def load_init(model,path):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    state=obj.get('model_state',obj.get('state_dict',obj))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    # Accept torchvision ResNet weights or this project's state dictionaries.
    if 'layer1.0.conv1.weight' in state:
        mapped={}
        for k,v in state.items():
            if k.startswith('fc.'): continue
            for i in range(1,5): k=k.replace(f'layer{i}.',f'layers.{i-1}.')
            mapped['backbone.'+k]=v
        state=mapped
    expected_r101=len(model.backbone.layers[2])==23
    actual_r101=any(k.startswith('backbone.layers.2.22.') for k in state)
    if expected_r101!=actual_r101: raise ValueError('Checkpoint backbone depth differs (R50/R101); do not silently partially load')
    own=model.state_dict(); usable={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
    required=[k for k in own if k.startswith('backbone.') and 'causal_attention' not in k]
    missing=[k for k in required if k not in usable]
    if missing: raise ValueError(f'Incomplete backbone initialization, e.g. {missing[:3]}')
    result=model.load_state_dict(usable,strict=False)
    return {'loaded_keys':len(usable),'missing_keys':result.missing_keys,'unused_keys':[k for k in state if k not in usable]}

def load_trained(path,device):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if not all(k in obj for k in ('config','variant','model_state')): raise ValueError('Use a checkpoint produced by the reviewed train.py')
    model=make_model(obj['config'],obj['variant'],False)
    model.load_state_dict(obj['model_state'],strict=True)
    return model.to(device).eval(),obj
