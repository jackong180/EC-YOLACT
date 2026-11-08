"""COCO loader: all images, crowd last, deterministic validation."""
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import preprocess_input

class COCODetection(Dataset):
    def __init__(self, image_path, coco, COCO_LABEL_MAP, augmentation=None, input_shape=(544,544)):
        self.image_path=Path(image_path); self.coco=coco; self.ids=sorted(coco.getImgIds())
        self.label_map=COCO_LABEL_MAP; self.augmentation=augmentation; self.input_shape=input_shape
    def __len__(self): return len(self.ids)
    def __getitem__(self,index):
        iid=self.ids[index]; meta=self.coco.loadImgs(iid)[0]
        image=np.asarray(Image.open(self.image_path/meta['file_name']).convert('RGB'),dtype=np.float32)
        h,w=image.shape[:2]
        if (w,h)!=(meta['width'],meta['height']): raise ValueError(f'Image/annotation size mismatch: {iid}')
        anns=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[iid]))
        anns=[a for a in anns if a['bbox'][2]>0 and a['bbox'][3]>0]
        anns.sort(key=lambda a:int(a.get('iscrowd',0)))
        crowds=sum(bool(a.get('iscrowd',0)) for a in anns)
        boxes=np.array([[a['bbox'][0]/w,a['bbox'][1]/h,(a['bbox'][0]+a['bbox'][2])/w,(a['bbox'][1]+a['bbox'][3])/h,self.label_map[a['category_id']]-1] for a in anns],dtype=np.float32).reshape(-1,5)
        masks=np.array([self.coco.annToMask(a) for a in anns],dtype=np.float32).reshape(-1,h,w)
        if self.augmentation is not None and len(anns)>crowds:
            image,masks,b,lab=self.augmentation(image,masks,boxes[:,:4],{'num_crowds':crowds,'labels':boxes[:,4]})
            boxes=np.column_stack((b,lab['labels'])).astype(np.float32); crowds=lab['num_crowds']
        else:
            oh,ow=self.input_shape; image=cv2.resize(image,(ow,oh))
            masks=np.array([cv2.resize(m,(ow,oh),interpolation=cv2.INTER_NEAREST) for m in masks],dtype=np.float32).reshape(-1,oh,ow)
        return preprocess_input(image).transpose(2,0,1).astype(np.float32),boxes,masks,crowds

def dataset_collate(batch):
    im,box,mask,crowd=zip(*batch)
    return torch.from_numpy(np.stack(im)),[torch.from_numpy(x) for x in box],[torch.from_numpy(x) for x in mask],list(crowd)
